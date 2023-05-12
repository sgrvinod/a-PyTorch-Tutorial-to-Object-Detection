import time
import pandas as pd
from copy import copy
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms.v2 as v2
from torchvision import disable_beta_transforms_warning
# from models.SSD300 import SSD300, MultiBoxLoss
from models.EfficientNetSSD300 import EfficientNetSSD300, MultiBoxLoss
from datasets import SerengetiDataset, get_dataset_params
from transformations import BBoxRandomCrop, BBoxRandomHorizontalFlip, BBoxResize, BBoxToFractional
from utils import *

# Disable torchvision warnings
disable_beta_transforms_warning()

# Data parameters
data_folder = './SSDataset'  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
with open('snapshot-serengeti/classes.csv', 'r') as f:
    classes = pd.read_csv(f)
n_classes = len(classes)  # number of different types of objects

label_map = {}
for i, row in classes.iterrows():
    label_map[i] = row['name']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
batch_size = 32  # batch size
iterations = 120_000  # number of iterations to train
workers = 1 # number of workers for loading data in the DataLoader
print_freq = 1  # print training status every __ batches
lr = 1e-3  # learning rate
decay_lr_at = [80_000, 100_000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True

def main():
    """
    Training.
    """
    disable_beta_transforms_warning()
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        model = EfficientNetSSD300(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(anchor_boxes=model.anchor_boxes).to(device)
    print(f'\nLoaded model to {device}.')

    # Custom dataloaders
    train_dataset, _ = get_train_val_datasets()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here
    
    print(f'Initialized data loader.')


    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    epochs = iterations // (len(train_dataset) // batch_size)
    decay_lr_at = [it // (len(train_dataset) // batch_size) for it in decay_lr_at]

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # Save checkpoint
        # save_checkpoint(epoch, model, optimizer) REPLACE THIS
        state = {'epoch': epoch, 'model': model, 'optimizer': optimizer}
        filename = 'checkpoint_efficientnet_ssd300_2.pth.tar'
        torch.save(state, filename)

def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    print(f'Beginning epoch {epoch}.')
    for i, (images, boxes, labels) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
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
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored

def get_train_val_datasets():
    disable_beta_transforms_warning()
    params = get_dataset_params()
    dataset = SerengetiDataset(*params)

    train_split = 0.7
    n_train = int(len(dataset) * train_split)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, (n_train, len(dataset)-n_train),)
    
    train_transform = v2.Compose([
        BBoxToFractional(),
        BBoxRandomHorizontalFlip(),
        BBoxRandomCrop((0.7,1.0), (0.9,1.1)),
        BBoxResize((300, 300)),
        v2.ColorJitter(brightness=0.1, contrast=0.05),
        v2.Compose([v2.ToImageTensor(), v2.ConvertImageDtype()]),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = v2.Compose([
        BBoxToFractional(),
        BBoxResize((300, 300)),
        v2.Compose([v2.ToImageTensor(), v2.ConvertImageDtype()]),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset.dataset = copy(dataset)
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    return train_dataset, val_dataset


if __name__ == '__main__':
    main()
