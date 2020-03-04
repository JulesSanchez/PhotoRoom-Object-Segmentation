"""Data augmentation pipelines."""
import os
import pandas as pd
import cv2
import torch
import numpy as np
from torch.utils import data
from albumentations import Compose, Resize, RandomResizedCrop, HorizontalFlip, Rotate
from albumentations import OneOf, GaussNoise, GaussianBlur, RGBShift
from albumentations import RandomBrightnessContrast, Normalize, Resize
from albumentations.pytorch import ToTensor
from sklearn.utils import shuffle

TRAIN_NAME = 'train_ids_duc.csv'
VAL_NAME = 'val_ids_duc.csv'
PATH = 'data/train'

TRAIN_NAME = 'train_ids_duc.csv'
VAL_NAME = 'val_ids_duc.csv'
PATH = 'data/train'

train_transform = Compose([
    OneOf([
        RandomResizedCrop(224, 224, scale=(0.75, 1.0), p=0.2),
        Resize(224, 224, p=0.8),
    ], p=1.0),
    OneOf([
        GaussNoise(p=0.5),
        GaussianBlur(p=0.5)
    ], p=0.4),
    HorizontalFlip(p=0.5),
    Rotate(15, p=0.9),
    OneOf([
        RGBShift(p=0.5),
        RandomBrightnessContrast(brightness_limit=0.4,
                                 contrast_limit=0.3, p=0.5),
    ], p=0.5),
    Normalize(),
    ToTensor()
])

val_transforms = Compose([
    Resize(224, 224),
    Normalize(),
    ToTensor()
])


class DataLoaderSegmentation(data.Dataset):
    def __init__(self, folder_path, batch_size, csv_name, transforms=ToTensor()):
        super(DataLoaderSegmentation, self).__init__()
        self.img_files = list(pd.read_csv(
            os.path.join(folder_path, csv_name))['img'])
        self.img_files = [os.path.join(folder_path, 'images', os.path.basename(
            img_path) + '.jpg') for img_path in self.img_files]
        self.mask_files = []
        self.transforms = transforms
        for img_path in self.img_files:
            self.mask_files.append(os.path.join(
                folder_path, 'masks', os.path.basename(img_path)[:-4] + '.png'))
        self.batch_size = batch_size
        self.N = len(self.img_files)//batch_size + (len(self.img_files) % batch_size > 0)

    def __getitem__(self, index):
        imgs = []
        masks = []
        for idx in range(self.batch_size*index, self.batch_size*(index+1)):
            img_path = self.img_files[idx]
            mask_path = self.mask_files[idx]
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if self.transforms is not None:
                augmented_ = self.transforms(
                    image=img, mask=mask)
                img = augmented_['image']
                mask = augmented_['mask'].long()
            imgs.append(img)
            masks.append(mask)
        if index == (len(self) - 1):
            self.img_files, self.mask_files = shuffle(self.img_files, self.mask_files)
        return torch.stack(imgs), torch.stack(masks).view(-1,*imgs[0].shape[1:])

    def __len__(self):
        return self.N


def plot_prediction(img: torch.Tensor, pred_mask: torch.Tensor, target: torch.Tensor = None):
    img = np.transpose(img.data.cpu().numpy(), axes=(1,2,0))
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])

    pred_mask = pred_mask.data.cpu().numpy()
    target = target.data.cpu().numpy()

    import matplotlib.pyplot as plt
    from typing import List
    if target is not None:
        num_plots = 3
    else:
        num_plots = 2
    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots + 1, 4), dpi=50)
    axes: List[plt.Axes]
    axes[0].imshow(img)
    axes[0].set_title("Original image")
    
    axes[1].imshow(pred_mask)
    axes[1].set_title("Predicted probabilities")
    
    if target is not None:
        axes[2].imshow(target)
        axes[2].set_title("Target mask")
    return fig


if __name__ == "__main__":
    train_dataset = DataLoaderSegmentation('data/train', 2, TRAIN_NAME,
                                           transforms=train_transform)
    
    imgs, masks = train_dataset[10]

    img = imgs[0].numpy()
    img = np.transpose(img, axes=(1,2,0))
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    mask = masks[0].numpy()
    
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9,4))
    plt.subplot(121)
    plt.imshow(img[...,::-1])
    plt.title("Image")
    
    plt.subplot(122)
    plt.imshow(mask, cmap="gray")
    plt.title("Mask")
    plt.tight_layout()
    plt.show()
