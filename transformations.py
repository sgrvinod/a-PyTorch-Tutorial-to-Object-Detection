import torch
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as v2
import random
import numpy as np
from copy import copy

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

class BBoxRandomCrop(object):
    def __init__(self, scale, ratio, retry=5):
        assert isinstance(scale, tuple)
        self.scale = scale
        assert isinstance(ratio, tuple)
        self.ratio = ratio
        assert isinstance(retry, int)
        self.retry = retry

    def __call__(self, sample):
        success = False
        attempts = 0
        image, boxes, labels = copy(sample)

        while not success and attempts < self.retry:
            # Randomly determine the scale and ratio of the crop
            scale = random.uniform(self.scale[0], self.scale[1])
            ratio = random.uniform(self.ratio[0], self.ratio[1])

            # Get the height and width of the input image
            h, w = image.height, image.width

            # Calculate the maximum crop dimensions
            cropped_h_max = h
            cropped_w_max = w
            if ratio > 0:
                cropped_w_max = int(cropped_h_max / ratio)
            if cropped_w_max > w:
                cropped_w_max = w
                cropped_h_max = int(cropped_w_max * ratio)

            # Calculate the actual crop dimensions based on the scale
            area_ratio = np.sqrt(scale)
            cropped_h = int(cropped_h_max * area_ratio)
            cropped_w = int(cropped_w_max * area_ratio)

            # Randomly determine the top-left coordinates of the crop
            top = np.random.randint(0, h - cropped_h) if h > cropped_h else 0
            left = np.random.randint(0, w - cropped_w) if w > cropped_w else 0

            # Calculate the bottom-right coordinates of the crop
            right = min(w, left + cropped_w)
            bottom = min(h, top + cropped_h)


            updated_boxes = torch.Tensor(len(boxes), 4)
            # If the center is off screen, set it to empty
            for i, box in enumerate(boxes):
                updated_boxes[i] = self.crop_bbox(box, (top, left))

            updated_labels = labels
            for i, (x,y,_,_) in enumerate(boxes):
                if x <= 0 or y <= 0 or x >= cropped_w or y >= cropped_h:
                    updated_labels[i] = 0

            updated_boxes = updated_boxes[updated_labels != 0]
            updated_labels = updated_labels[updated_labels != 0]
            
            if len(updated_boxes) > 0:
                success = True
                image = TF.to_tensor(image)
                image = image[:, top: bottom, left: right]
                image = TF.to_pil_image(image)
                labels = updated_labels
                boxes = updated_boxes
            else:
                image, boxes, labels = copy(sample)
                attempts += 1

        return image, boxes, labels
    
    def crop_bbox(self, bbox, crop):
        top, left = crop
        x, y, w, h = copy(bbox)
        x = bbox[0] - left
        y = bbox[1] - top
        return torch.Tensor([x,y,w,h])

class BBoxRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        assert isinstance(p, float)
        self.p = p

    def __call__(self, sample):
        image, bboxes, labels = sample
        flip = random.random() <= self.p
        if flip:
            image = TF.hflip(image)
            bboxes = [self.bbox_hflip(image.width, bbox) for bbox in bboxes]
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
        image, bboxes, labels = sample
        h, w = image.height, image.width
        new_h, new_w = self.output_size
        image = v2.Resize((new_h, new_w))(image)
        for i, bbox in enumerate(bboxes):
            x_scale = new_w / w
            y_scale = new_h / h
            bboxes[i][0] *= x_scale
            bboxes[i][1] *= y_scale
            bboxes[i][2] *= x_scale
            bboxes[i][3] *= y_scale
        return image, bboxes, labels
