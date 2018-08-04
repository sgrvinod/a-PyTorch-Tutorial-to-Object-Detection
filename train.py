import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *

# Data parameters
data_folder = './'

# Model parameters
n_classes = len(label_map)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training parameters
checkpoint = None
pretrained_base = './VGG16_ILSVRC_CLS_LOC_converted_degroot.pth.tar'
batch_size = 32
start_epoch = 0
epochs = 200
epochs_since_improvement = 0
best_loss = 100.
workers = 4
print_freq = 50
lr = 1e-3
momentum = 0.9
weight_decay = 5e-4

cudnn.benchmark = True


def main():
    global epochs_since_improvement, start_epoch, label_map, best_loss, epoch, checkpoint

    if checkpoint is None:
        model = SSD300(n_classes=n_classes)
        model.init_conv2d()
        model.load_pretrained_base(pretrained_base)
        optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_loss = checkpoint['best_loss']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    model = model.to(device)

    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Custom dataloaders
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=False)
    val_dataset = PascalVOCDataset(data_folder,
                                   split='test',
                                   keep_difficult=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                             collate_fn=val_dataset.collate_fn, num_workers=workers, pin_memory=True)

    for epoch in range(start_epoch, epochs):

        # if epochs_since_improvement == 30:
        #     break
        if epoch in [154, 193, 231]:
            adjust_learning_rate(optimizer, 0.1)

        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        val_loss = validate(val_loader=val_loader,
                            model=model,
                            criterion=criterion)

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, val_loss, best_loss, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        predicted_locs, predicted_scores = model(images)

        loss = criterion(predicted_locs, predicted_scores, boxes, labels)

        optimizer.zero_grad()
        loss.backward()
        # clip_gradient(optimizer, 5.)
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    del images, boxes, labels


def validate(val_loader, model, criterion):
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    # det_boxes = list()
    # det_labels = list()
    # det_scores = list()
    # true_boxes = list()
    # true_labels = list()
    # true_difficulties = list()

    with torch.no_grad():
        for i, (images, boxes, labels, difficulties) in enumerate(val_loader):

            images = images.to(device)

            predicted_locs, predicted_scores = model(images)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            loss = criterion(predicted_locs, predicted_scores, boxes, labels)

            # det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores)
            #
            # det_boxes.extend(det_boxes_batch)
            # det_labels.extend(det_labels_batch)
            # det_scores.extend(det_scores_batch)
            # true_boxes.extend(boxes)
            # true_labels.extend(labels)
            # true_difficulties.extend(difficulties)

            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i % print_freq == 0:
                print('[{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses))

    # mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

    print('\n * LOSS - {loss.avg:.3f}\n'.format(loss=losses))

    return losses.avg


if __name__ == '__main__':
    main()
