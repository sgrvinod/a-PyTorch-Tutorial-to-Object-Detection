import numpy as np
import pandas as pd
import random
import json
import os
from copy import copy
from PIL import Image
from io import BytesIO
from retry import retry

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms.v2 import ToTensor
from torch.utils.data import Dataset

from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from requests.exceptions import SSLError

from utils import transform
from credentials import get_creds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SerengetiBBoxDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, root, images_path, bboxes_path, classes_path, seasons=None, transform=ToTensor()):
        '''
        self.version = {DAY, NIGHT, COMBINED}

        '''

        self.root = root
        self.transform = transform
        self.seasons = seasons
        self.gcs_service = build('storage', 'v1', credentials=get_creds())

        images_df = pd.read_csv(images_path)
        
        with open(bboxes_path, 'r') as f:
            self.bbox_objects = json.load(f)['annotations']
        
        with open(classes_path, 'r') as f:
            class_objects = json.load(f)['categories']
        
        self.classes = {obj['name'].lower(): obj['id'] for obj in class_objects}
        self.images = images_df['image_path_rel'].tolist()
        if self.seasons:
            self.images = list(filter(self.check_seasons, self.images)) # filter out seasons
        self.labels = []
        self.bboxes = [[] for _ in range(len(self.images))]

        for species in images_df['question__species'].tolist():
            species = species.lower()
            if species == 'blank':
                species = 'empty'
            if species == 'vervetmonkey':
                species = 'monkeyvervet'
            self.labels.append(self.classes[species])
        
        image_dict = {filename: i for i, filename in enumerate(self.images)}

        for obj in self.bbox_objects:
            image = obj['image_id'] + '.JPG'
            if image in image_dict:
                bbox = obj['bbox']
                index = image_dict[image]
                self.bboxes[index].append(bbox)

        i = 0
        while i < len(self.bboxes):
            if len(self.bboxes[i]) == 0:
                del self.images[i]
                del self.bboxes[i]
            else:
                i += 1
        
    def __getitem__(self, i):
        # Read Image
        path = os.path.join(self.root, self.images[i])
        # image = Image.open(path, mode='r')
        image = self.request_image_from_gcs(path)
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels)
        boxes = torch.FloatTensor(self.bboxes[i])               # (n_objects, 4)
        labels = torch.LongTensor([self.labels[i]]*len(boxes))  # (n_objects), all objects are same label

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)

        return image, boxes, labels

    @retry(exceptions=SSLError, tries=10, delay=0.1)
    def request_image_from_gcs(self, image_path):
        bucket_name = 'public-datasets-lila'
        image_bytes = BytesIO()
        request = self.gcs_service.objects().get_media(bucket=bucket_name, object=image_path)
        media = MediaIoBaseDownload(image_bytes, request)

        done = False
        while not done:
            _, done = media.next_chunk()

        image_bytes.seek(0)

        return Image.open(image_bytes)

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        return images, boxes, labels  # tensor (N, 3, x, y), 3 lists of N tensors each

    def check_seasons(self, image):
        for season in self.seasons:
            if season + '/' in image:
                return True
        return False


# Reference Dataset
'''
class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transformations
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

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

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each
'''