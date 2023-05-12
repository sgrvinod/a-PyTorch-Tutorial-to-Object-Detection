import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
from ast import literal_eval

class SerengetiDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """
    def __init__(self, image_folder, images_df, annotations_df, classes_df, transform=None):
        self.image_folder = image_folder
        self.images_df = images_df
        self.annotations_df = annotations_df
        self.classes_df = classes_df
        self.transform = transform

        self.bboxes = {row['id']: [] for _, row in self.images_df.iterrows()} # 4909 images have no bounding boxes, filter these in df 
        for i, row in self.annotations_df.iterrows():                         # images have been filtered, now boxes need to be filtered?
            self.bboxes[row['image_id']].append(i)

        self.annotations_df['bbox'] = self.annotations_df['bbox'].apply(literal_eval)
        
        print('Initialized dataset.')

    def __getitem__(self, i):
        image_info = self.images_df.iloc[i]

        path = os.path.join(self.image_folder, image_info['image_path_rel'])
        image = Image.open(path)

        box_idxs = self.bboxes[image_info['id']]
        boxes = torch.FloatTensor([self.annotations_df.iloc[i]['bbox'] for i in box_idxs])

        species = image_info['question__species'].lower()
        label_step = self.classes_df.loc[self.classes_df['name'] == species, 'id']
        label = self.classes_df.loc[self.classes_df['name'] == species, 'id'].iloc[0] #####
        labels = torch.FloatTensor([label for _ in boxes])
        
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)

        return image, boxes, labels

    def __len__(self):
        return len(self.images_df)

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

def get_dataset_params():
    image_folder = '../snapshot-serengeti'
    images_df = pd.read_csv('./snapshot-serengeti/bbox_images_non_empty_downloaded.csv')
    annotations_df = pd.read_csv('./snapshot-serengeti/bbox_annotations_downloaded.csv')
    classes_df = pd.read_csv('./snapshot-serengeti/classes.csv')

    return image_folder, images_df, annotations_df, classes_df

from collections import Counter
def main():
    dataset = SerengetiBBoxDataset(*get_dataset_params())
    print('Serengeti Dataset')
    print(f'num_items: {len(dataset)}')
    print(f'item_0: {dataset[0]}')
    c = Counter([len(boxes) for boxes in dataset.bboxes.values()])
    print(f'boxes_per_img: {c}')

if __name__ == '__main__':
    main()

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