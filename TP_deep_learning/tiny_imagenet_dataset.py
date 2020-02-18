import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import cv2
import sys

nb_classes = 20
nb_samples_per_class = 500
nb_samples_per_class_train = 400
nb_samples_per_class_test = 50
nb_samples_per_class_validation = 50

class TinyImageNetDatasetTrain(Dataset):
    def __init__(self, img_dir):
        super(TinyImageNetDatasetTrain, self).__init__()
        self.img_dir = img_dir

    def __len__(self):
        return nb_classes*nb_samples_per_class_train

    def __getitem__(self, idx):
        idx_class = idx//nb_samples_per_class_train
        idx_sample = idx%nb_samples_per_class_train

        img_path = self.img_dir + '/{:03}/{:03}.jpg'.format(idx_class, idx_sample)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

        img = torch.tensor(np.array(img.transpose([2, 0, 1])))
        id_class = torch.tensor(idx_class).type(torch.int64)
        return img, id_class

class TinyImageNetDatasetTest(Dataset):
    def __init__(self, img_dir):
        super(TinyImageNetDatasetTest, self).__init__()
        self.img_dir = img_dir

    def __len__(self):
        return nb_classes * nb_samples_per_class_test

    def __getitem__(self, idx):
        idx_class = idx // nb_samples_per_class_test
        idx_sample = idx % nb_samples_per_class_test

        img_path = self.img_dir + '/{:03}/{:03}.jpg'.format(idx_class, nb_samples_per_class_train + nb_samples_per_class_test + idx_sample)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

        img = torch.tensor(np.array(img.transpose([2, 0, 1])))
        id_class = torch.tensor(idx_class).type(torch.int64)
        return img, id_class

class TinyImageNetDatasetValidation(Dataset):
    def __init__(self, img_dir):
        super(TinyImageNetDatasetValidation, self).__init__()
        self.img_dir = img_dir

    def __len__(self):
        return nb_classes * nb_samples_per_class_validation

    def __getitem__(self, idx):
        idx_class = idx // nb_samples_per_class_validation
        idx_sample = idx % nb_samples_per_class_validation

        img_path = self.img_dir + '/{:03}/{:03}.jpg'.format(idx_class, nb_samples_per_class_train + idx_sample)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

        img = torch.tensor(np.array(img.transpose([2, 0, 1])))
        id_class = torch.tensor(idx_class).type(torch.int64)
        return img, id_class