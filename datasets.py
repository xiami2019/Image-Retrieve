#Author by Cqy2019

import torch
import os
import numpy as np

from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from scipy.io import loadmat

class CUB_200_2011(Dataset):
    '''
    CUB_200_2011 Dataset for image retrieval
    '''
    def __init__(self, root, if_train, if_database=False):
        '''
        file: data root.
        if_train: to identify train set of test set.
        '''
        self.root = root
        self.if_train = if_train
        self.if_database = if_database
        self.train_index = []
        self.test_index = []
        self.index2imgname = []
        self.index2label = []

        with open(os.path.join(root, 'train_test_split.txt'), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = list(line.strip('\n').split())
                if int(line[1]) == 1:
                    self.train_index.append(int(line[0]) - 1)
                else:
                    self.test_index.append(int(line[0]) - 1)
        
        with open(os.path.join(root, 'images.txt'), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = list(line.strip('\n').split())
                self.index2imgname.append(line[1])
        
        with open(os.path.join(root, 'image_class_labels.txt'), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = list(line.strip('\n').split())
                self.index2label.append(int(line[1]) - 1)
        
        #check 5994 5794
        assert len(self.train_index) == 5994
        assert len(self.test_index) == 5794
        assert len(self.index2label) == len(self.index2imgname) == 5994 + 5794

        # transforms.RandomPerspective(),
        # transforms.RandomGrayscale(),

        if self.if_train is True and self.if_database is False:
            self.transforms = transforms.Compose([
                            transforms.Resize(256),
                            transforms.RandomCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
        else:
            self.transforms = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])

    def __len__(self):
        if self.if_train:
            return len(self.train_index)
        else:
            return len(self.test_index)

    def __getitem__(self, idx):
        if self.if_train:
            imagename = self.index2imgname[self.train_index[idx]]
            label = self.index2label[self.train_index[idx]]
        else:
            imagename = self.index2imgname[self.test_index[idx]]
            label = self.index2label[self.test_index[idx]]
        image = Image.open(os.path.join(self.root, 'images', imagename)).convert('RGB')
        image = self.transforms(image)

        return image, label

class Stanford_Dog(Dataset):
    '''
    Stanford_Dog Dataset for image retrieval
    '''
    def __init__(self, root, if_train, if_database=False):
        '''
        file: data root.
        if_train: to identify train set of test set.
        '''
        self.root = root
        self.if_train = if_train
        self.if_database = if_database
        
        if self.if_train:
            self.images = [image[0][0] for image in loadmat(os.path.join(root, 'train_list.mat'))['file_list']]
            self.labels = [(int(image[0]) - 1) for image in loadmat(os.path.join(root, 'train_list.mat'))['labels']]
        else:
            self.images = [image[0][0] for image in loadmat(os.path.join(root, 'test_list.mat'))['file_list']]
            self.labels = [(int(image[0]) - 1) for image in loadmat(os.path.join(root, 'test_list.mat'))['labels']]
        
        if self.if_train:
            assert len(self.images) == len(self.labels) == 12000
        else:
            assert len(self.images) == len(self.labels) == 8580

        if self.if_train is True and self.if_database is False:
            self.transforms = transforms.Compose([
                            transforms.Resize(256),
                            transforms.RandomCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
        else:
            self.transforms = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        imagename = self.images[idx]
        label = self.labels[idx]
        image = Image.open(os.path.join(self.root, 'Images', imagename)).convert('RGB')
        image = self.transforms(image)

        return image, label

if __name__ == '__main__':
    train_set = Stanford_Dog('datasets/Stanford_Dogs', True)
    print(len(train_set))
    for i in range(101):
        image, label = train_set[i]
        print(image.size())
        print(label)