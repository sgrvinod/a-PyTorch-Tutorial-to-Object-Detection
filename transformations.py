import torch
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as v2
from torchvision.transforms import ToPILImage, ToTensor
import torch.nn.functional as F
import random
import numpy as np
from copy import copy
from utils import find_jaccard_overlap


class BBoxToFractional(object):
    def __call__(self, sample):
        """
            returns: new bounding box in fractional format (ie cx cy w h)
        """
        image, boxes, labels = sample
        fractional_boxes = torch.FloatTensor(len(boxes), 4)
        for i, box in enumerate(boxes):
            x, y, w, h = box
            fractional_boxes[i] = torch.Tensor([
                x + (w / 2), 
                y + (w / 2), 
                w / 2, 
                h / 2
            ])

        return image, fractional_boxes, labels


class BBoxToBoundary(object):
    def __call__(self, sample):
        """
            returns: new bounding box in boundary format (ie x y w h)
        """
        image, boxes, labels = sample
        fractional_boxes = torch.FloatTensor(len(boxes), 4)
        for i, box in enumerate(boxes):
            x, y, w, h = box
            fractional_boxes[i] = torch.Tensor([
                x - w, 
                y - w, 
                w * 2,
                h * 2
            ])

        return image, fractional_boxes, labels


class BBoxRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        assert isinstance(p, float)
        self.p = p

    def __call__(self, sample):
        '''
        sample: image, boxes, labels
        image:  tensor (3, h, w)
        boxes:  tensor (4, n)
        labels: tensor (n)
        output: tensor, tensor, tensor
        '''
        image, bboxes, labels = sample
        flip = random.random() <= self.p
        if flip:
            image = TF.hflip(image)
            bboxes = [self.bbox_hflip(image.size(2), bbox) for bbox in bboxes]
            bboxes = torch.stack(bboxes)
        return image, bboxes, labels


    def bbox_hflip(self, width, bbox):
        bbox[0] = width - bbox[0]
        return bbox


class BBoxResize(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        '''
        sample: image, boxes, labels
        image:  tensor (3, h, w)
        boxes:  tensor (4, n)
        labels: tensor (n)
        output: PIL Image, tensor, tensor
        '''
        image, bboxes, labels = sample
        h, w = image.size(1), image.size(2)
        resized_image = F.interpolate(image.unsqueeze(0), size=self.output_size, mode='bilinear', align_corners=False).squeeze(0)
        x_scale = self.output_size[0] / w
        y_scale = self.output_size[1] / h
        scales = torch.FloatTensor([x_scale, y_scale, x_scale, y_scale]).unsqueeze(0)
        bboxes = bboxes * scales
        return resized_image, bboxes, labels


class BBoxRandomCrop(object):
    def __init__(self, scale, ratio, retry=5):
        assert isinstance(scale, tuple)
        self.scale = scale
        assert isinstance(ratio, tuple)
        self.ratio = ratio
        assert isinstance(retry, int)
        self.retry = retry

    def __call__(self, sample):
        '''
        image: a tensor (3 x h x w)
        boxes: a tensor (n x 4) in boundary form
        '''
        max_trials = 50
        image, boxes, labels = copy(sample)
        original_h, original_w = image.size(1), image.size(2)

        while True:
            min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])

            if min_overlap is None:
                return image, boxes, labels

            for _ in range(max_trials):
                min_scale = self.scale[0]
                scale_h = random.uniform(min_scale, 1)
                scale_w = random.uniform(min_scale, 1)
                new_h = int(scale_h * original_h)
                new_w = int(scale_w * original_w)

                # Aspect ratio has to be in [0.5, 2]
                aspect_ratio = new_h / new_w
                if not self.ratio[0] < aspect_ratio < self.ratio[1]:
                    continue

                left = random.randint(0, original_w - new_w)
                right = left + new_w
                top = random.randint(0, original_h - new_h)
                bottom = top + new_h
                crop = torch.FloatTensor([left, top, right, bottom])

                overlap = find_jaccard_overlap(crop.unsqueeze(0), boxes).squeeze(0)
                if overlap.max().item() < min_overlap:
                    continue
                
                new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)

                # Find centers of original bounding boxes
                bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.  # (n_objects, 2)

                # Find bounding boxes whose centers are in the crop
                centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                        bb_centers[:, 1] < bottom)  # (n_objects), a Torch uInt8/Byte tensor, can be used as a boolean index

                # If not a single bounding box has its center in the crop, try again
                if not centers_in_crop.any():
                    continue

                # Discard bounding boxes that don't meet this criterion
                new_boxes = boxes[centers_in_crop, :]
                new_labels = labels[centers_in_crop]
                new_difficulties = difficulties[centers_in_crop]

                # Calculate bounding boxes' new coordinates in the crop
                new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])  # crop[:2] is [left, top]
                new_boxes[:, :2] -= crop[:2]
                new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])  # crop[2:] is [right, bottom]
                new_boxes[:, 2:] -= crop[:2]

                return new_image, new_boxes, new_labels


def train_transform():
    return v2.Compose([
        v2.Compose([v2.ToImageTensor(), v2.ConvertImageDtype()]),
        BBoxRandomHorizontalFlip(),
        BBoxRandomCrop((0.7,1.0), (0.9,1.1)),
        BBoxResize(300),
        BBoxToFractional(),
        v2.ColorJitter(brightness=0.1, contrast=0.05),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])