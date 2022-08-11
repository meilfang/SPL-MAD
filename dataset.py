import os
import pandas as pd
import numpy as np
import cv2
from collections import defaultdict

import torch
from torch.utils.data import Dataset

import albumentations
from albumentations.pytorch import ToTensorV2

PRE__MEAN = [0.5, 0.5, 0.5]
PRE__STD = [0.5, 0.5, 0.5]

class TrainDataset(Dataset):

    def __init__(self, csv_file, input_shape=(224, 224)):
        self.dataframe = pd.read_csv(csv_file)
        self.composed_transformations = albumentations.Compose([
            albumentations.SmallestMaxSize(max_size=input_shape[0]),
            albumentations.RandomCrop(height=input_shape[0], width=input_shape[0]),
            albumentations.HorizontalFlip(),
            albumentations.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0.1, p=0.5),
            albumentations.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            albumentations.Normalize(PRE__MEAN, PRE__STD, always_apply=True),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        img_path = self.dataframe.iloc[idx, 0]
        if '/data/fboutros' in img_path:
            img_path = img_path.replace('/data/fboutros', '/data/mfang/FR_DB')

        image = cv2.imread(img_path)
        image = self.composed_transformations(image = image)['image']

        return {
            "images": image,
        }

class TestDataset(Dataset):

    def __init__(self, csv_file, input_shape=(224, 224)):
        #self.image_dir = image_dir
        self.dataframe = pd.read_csv(csv_file)
        self.composed_transformations = albumentations.Compose([
            albumentations.SmallestMaxSize(max_size=input_shape[0]),
            albumentations.CenterCrop(height=input_shape[0], width=input_shape[0]),
            albumentations.Normalize(PRE__MEAN, PRE__STD, always_apply=True),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label_str = self.dataframe.iloc[idx, 1]
        image = cv2.imread(img_path)
        label = 0 if label_str == 'bonafide' else 1

        image = self.composed_transformations(image=image)['image']

        return {
            "images": image,
            "labels": torch.tensor(label, dtype = torch.float),
            "img_path": img_path
        }
