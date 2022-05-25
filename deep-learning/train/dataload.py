import torch
import numpy as np
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

txtpath = r'E:\dataset\pic\result\train.txt'
datapath = r'E:\dataset\pic\result'

class MyDataset(Dataset):
    def __init__(self, txtpath, transform=None):
        imgs = []
        label = []
        datainfo = open(txtpath, 'r')
        for line in datainfo:
            line = line.strip('\n')
            if line == 'train.bat' or line == 'train.txt':
                continue
            words = line.split('_')
            imgs.append(line)
            label.append(words[0])
        self.imgs = imgs
        self.label = label
        assert len(self.imgs) == len(self.label)
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        pic = self.imgs[index]
        label = self.label[index]
        pic = Image.open(datapath + '\\' + pic)
        if self.transform is not None:
            pic = self.transform(pic)
        label = np.array(label).astype(int)
        label = torch.from_numpy(label)
        return pic, label

