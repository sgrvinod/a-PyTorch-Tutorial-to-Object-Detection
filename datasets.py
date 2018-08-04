import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import Transform


class PascalVOCDataset(Dataset):

    def __init__(self, data_folder, split, keep_difficult=False):
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}
        if self.split == 'TEST':
            assert keep_difficult == True, 'MUST keep difficult boxes during val/test for mAP calculation!'

        self.data_folder = data_folder
        self.transform = Transform(split=self.split)
        self.keep_difficult = keep_difficult

        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)

        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        image, boxes, labels, difficulties = self.transform(image, boxes, labels, difficulties)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        images = list()
        boxes = list()
        labels = list()
        difficulties = list()
        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])
        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, ...), 3 lists of N tensors each
