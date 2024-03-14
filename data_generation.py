import torch
import os
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset

''' 数据读取 '''
def readfile(path, b, label):
    image_dir = sorted(os.listdir(path))
    # print(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))  # os.path.join(path, file) 路径名合并
        x[i, :, :] = cv2.resize(img, (128, 128))
        y[i]=b
    if label:
        return x, y
    else:
        return x


def data_generation():

    train_dir = 'data/train'
    test_dir = 'data/'
    print("---------------Reading data---------------")

    train_x0, train_y0 = readfile(os.path.join(train_dir, "0"), 0, True)
    train_x1, train_y1 = readfile(os.path.join(train_dir, "1"), 1, True)
    train_x2, train_y2 = readfile(os.path.join(train_dir, "2"), 2, True)
    train_x3, train_y3 = readfile(os.path.join(train_dir, "3"), 3, True)
    train_x = np.concatenate((train_x0, train_x1, train_x2, train_x3), axis=0)
    train_y = np.concatenate((train_y0, train_y1, train_y2, train_y3), axis=0)

    test_x = readfile(os.path.join(test_dir, "test"), 4, False)  # 测试集的标签定义为四表示测试集没有标签。

    return train_x, train_y, test_x


class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X


class ImageDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X