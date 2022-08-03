import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CamVidDataset(Dataset):

    def __init__(self, images_dir, labels_dir, transform=None, labels_transform=None):

        self.images_dir = images_dir
        self.labels_dir = labels_dir

        self.images = os.listdir(images_dir)
        self.labels = os.listdir(labels_dir)

        self.transform = transform
        self.labels_transform = labels_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_path = os.path.join(self.images_dir, self.images[idx])
        label_path = os.path.join(self.labels_dir, self.labels[idx])

        image = Image.open(image_path)
        label = Image.open(label_path)

        if self.transform:
            image = self.transform(image)
        if self.labels_transform:
            label = self.labels_transform(label)
        label = torch.as_tensor(np.array(label).transpose(2, 0, 1), dtype=torch.int64)

        return {"image": image, "label": label}


class CustomNormalizeTransform(object):

    def __call__(self, image):
        i_mean, i_std = image.mean([1, 2]), image.std([1, 2])

        i_transform = transforms.Compose([transforms.Normalize(i_mean, i_std)])

        n_image = i_transform(image)

        return n_image
