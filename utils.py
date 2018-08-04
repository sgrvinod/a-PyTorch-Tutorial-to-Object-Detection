import json
import os
import torch
import random
import xml.etree.ElementTree as ET
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0


def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()
    for object in root.iter('object'):

        difficult = int(object.find('difficult').text == '1')

        label = object.find('name').text.lower().strip()
        if label not in label_map:
            continue

        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}


def create_data_lists(voc07_path, voc12_path, output_folder):
    voc07_path = os.path.abspath(voc07_path)
    voc12_path = os.path.abspath(voc12_path)

    train_images = list()
    train_objects = list()
    n_objects = 0

    # Training data
    for path in [voc07_path, voc12_path]:

        with open(os.path.join(path, 'ImageSets/Main/trainval.txt')) as f:
            ids = f.read().splitlines()

        for id in ids:
            objects = parse_annotation(os.path.join(path, 'Annotations', id + '.xml'))
            if len(objects) == 0:
                continue
            n_objects += len(objects)
            train_objects.append(objects)
            train_images.append(os.path.join(path, 'JPEGImages', id + '.jpg'))

    assert len(train_objects) == len(train_images)

    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)

    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(train_images), n_objects, os.path.abspath(output_folder)))

    # Validation data
    test_images = list()
    test_objects = list()
    n_objects = 0

    with open(os.path.join(voc07_path, 'ImageSets/Main/test.txt')) as f:
        ids = f.read().splitlines()

    for id in ids:
        objects = parse_annotation(os.path.join(voc07_path, 'Annotations', id + '.xml'))
        if len(objects) == 0:
            continue
        test_objects.append(objects)
        n_objects += len(objects)
        test_images.append(os.path.join(voc07_path, 'JPEGImages', id + '.jpg'))

    assert len(test_objects) == len(test_images)

    with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
        json.dump(test_objects, j)

    print('\nThere are %d validation images containing a total of %d objects. Files have been saved to %s.' % (
        len(test_images), n_objects, os.path.abspath(output_folder)))


def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties):
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(
        true_labels) == len(
        true_difficulties)  # these are all lists of tensors of the same length, i.e. number of images
    n_classes = len(label_map)

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images)  # (n_objects), n_objects is the total no. of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # ignore difficult objects

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        true_positives = torch.zeros((n_class_detections), dtype=torch.float)  # (n_class_detections)
        false_positives = torch.zeros((n_class_detections), dtype=torch.float)  # (n_class_detections)
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class, and their difficulties
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), (), scalars
            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > 0.5:
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind] == 0:
                    true_positives[d] = 1
                else:
                    continue
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    print(average_precisions)
    mean_average_precision = average_precisions.mean().item()

    return mean_average_precision


def xy_to_cxcy(xy):
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def cxcy_to_xy(cxcy):
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (0.1 * priors_cxcy[:, 2:]),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) / 0.2], 1)  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    return torch.cat([gcxgcy[:, :2] * 0.1 * priors_cxcy[:, 2:] + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] * 0.2) * priors_cxcy[:, 2:]], 1)  # w, h


# Some augmentation functions below have been borrowed or adapted from
# From https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

def find_intersection(set_1, set_2):
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    intersection = find_intersection(set_1, set_2)  # (n1, n2)
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)
    return intersection / union  # (N1, N2)


class RandomCrop(object):
    def __init__(self):
        # Various crop-modes based on the minimum Jaccard overlap between the crop and any object/bounding box
        self.crop_modes = (0., .1, .3, .5, .7, .9, None)  # 'None' refers to no cropping

    def __call__(self, image, boxes, labels, difficulties):
        _, height, width = image.size()
        while True:
            # Randomly choose a mode
            mode = random.choice(self.crop_modes)

            # If not cropping
            if mode is None:
                return image, boxes, labels, difficulties

            # If cropping
            min_overlap = mode

            # Try up to 50 times
            for _ in range(50):
                new_image = image

                # Crop dimensions can be in [0.1, 1] of original dimensions
                w = random.uniform(0.1 * width, width)
                h = random.uniform(0.1 * height, height)

                # Aspect ratio to be in [0.5, 2]
                if h / w < 0.5 or h / w > 2:
                    continue

                # Crop coordinates (origin at top-left of image)
                left = int(random.uniform(0, width - w))
                right = int(left + w)
                top = int(random.uniform(0, height - h))
                bottom = int(top + h)
                crop = torch.FloatTensor([left, top, right, bottom])  # (4)

                # Calculate Jaccard overlap between the crop and the bounding boxes
                overlap = find_jaccard_overlap(crop.unsqueeze(0),
                                               boxes)  # (1, n_objects), n_objects is the no. of objects in this image
                overlap = overlap.squeeze(0)  # (n_objects)

                # If not a single bounding box has a Jaccard overlap of greater than the minimum, try again
                if overlap.max().item() < min_overlap:
                    continue

                # Crop image
                new_image = new_image[:, top:bottom, left:right]

                # Find centers of original bounding boxes
                bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # Find bounding boxes whose centers are in the crop
                centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                        bb_centers[:, 1] < bottom)

                # If not a single bounding box has its center in the crop, try again
                if not centers_in_crop.any():
                    continue

                # Discard bounding boxes that don't meet this criterion
                new_boxes = boxes[centers_in_crop, :]
                new_labels = labels[centers_in_crop]
                new_difficulties = difficulties[centers_in_crop]

                # Calculate bounding boxes' new coordinates in the crop
                new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])
                new_boxes[:, :2] -= crop[:2]
                new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])
                new_boxes[:, 2:] -= crop[:2]

                return new_image, new_boxes, new_labels, new_difficulties


class PhotometricDistort(object):
    def __init__(self):
        self.photo_distort = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)

    def __call__(self, image):
        new_image = self.photo_distort(image)

        return new_image


class Normalize(object):
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[123., 117., 104.],
                                              std=[1., 1., 1.])

    def __call__(self, image):
        new_image = self.normalize(image)

        return new_image


class HorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob
        self.flip = transforms.RandomHorizontalFlip(p=1.)

    def __call__(self, image, boxes):
        if random.random() > self.prob:
            return image, boxes

        # Flip image
        new_image = self.flip(image)

        # Flip boxes
        new_boxes = boxes
        new_boxes[:, 0] = image.width - boxes[:, 0]
        new_boxes[:, 2] = image.width - boxes[:, 2]
        new_boxes = new_boxes[:, [2, 1, 0, 3]]

        return new_image, new_boxes


class Resize(object):
    def __init__(self, dims=(300, 300), return_percent_coords=True):
        self.dims = dims
        self.return_percent_coords = return_percent_coords
        self.resize = transforms.Resize(dims)

    def __call__(self, image, boxes):
        # Resize image
        new_image = self.resize(image)

        # Resize bounding boxes
        old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
        new_boxes = boxes / old_dims  # percent coordinates

        if not self.return_percent_coords:
            new_dims = torch.FloatTensor([self.dims[1], self.dims[0], self.dims[1], self.dims[0]]).unsqueeze(0)
            new_boxes = new_boxes * new_dims

        return new_image, new_boxes


class Transform(object):
    def __init__(self, split):
        assert split in {'TRAIN', 'TEST'}
        self.split = split
        self.pd = PhotometricDistort()
        self.rc = RandomCrop()
        self.hf = HorizontalFlip()
        self.rs = Resize(return_percent_coords=True)
        self.nm = Normalize()
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def __call__(self, image, boxes, labels, difficulties):
        new_image = image
        new_boxes = boxes
        new_labels = labels
        new_difficulties = difficulties
        if self.split == 'TRAIN':
            new_image = self.pd(new_image)
            new_image = self.to_tensor(new_image)
            new_image, new_boxes, new_labels, new_difficulties = self.rc(new_image, new_boxes, new_labels,
                                                                         new_difficulties)
            new_image = self.to_pil(new_image)
            new_image, new_boxes = self.hf(new_image, new_boxes)
        new_image, new_boxes = self.rs(new_image, new_boxes)
        new_image = self.to_tensor(new_image) * 255.
        new_image = self.nm(new_image)

        return new_image, new_boxes, new_labels, new_difficulties


def adjust_learning_rate(optimizer, scale):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("DECAYING learning rate.\n The new LR is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, loss, best_loss, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': loss,
             'best_loss': best_loss,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoint_ssd300.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


if __name__ == '__main__':
    from PIL import Image

    image = Image.open('./manbike.png', 'r')
    image = image.convert('RGB')

    pd = PhotometricDistort()
    rc = RandomCrop()
    hf = HorizontalFlip()
    rs = Resize(return_percent_coords=False)
    nm = Normalize()
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    boxes = torch.FloatTensor([[237, 255, 427, 443], [465, 255, 654, 443]])
    labels = torch.LongTensor([0, 0])  # (n_objects)
    difficulties = torch.ByteTensor([0, 0])  # (n_objects)

    i, b, l, d = rc(to_tensor(image), boxes, labels, difficulties)
    to_pil(i).show()
