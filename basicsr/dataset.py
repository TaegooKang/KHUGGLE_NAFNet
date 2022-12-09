import torch
import torch.nn as nn
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.crops.transforms import RandomCrop
from glob import glob
from torch.utils.data import Dataset, DataLoader

class DenoisingData(Dataset):
    def __init__(self, root_data, mode):
        super().__init__()
        assert mode in ['train', 'valid'], "'train' or 'valid'"
        root_train = os.path.join(root_data, mode)
        root_gt = os.path.join(root_data, mode + '_gt')
        
        self.paths_train = sorted(glob(root_train+'/*.jpg'))
        self.paths_gt = sorted(glob(root_gt+'/*.jpg'))
        self.mode = mode
        
    def __getitem__(self, index):
        image = cv2.imread(self.paths_train[index])
        gt = cv2.imread(self.paths_gt[index])
        
        transform = A.Compose([
                RandomCrop(height=128, width=128),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
        ])
        totensor = A.Compose([
                ToTensorV2(), 
        ])
        # for training
        if self.mode == 'train':
            transformed = transform(image=image, mask=gt)
            image, gt = transformed['image'], transformed['mask']
        # for validation
        image = totensor(image=image)['image']
        gt = totensor(image=gt)['image']

        return image, gt

    def __len__(self):
        return len(self.paths_train)
    

if __name__ == '__main__':
    dataset = DenoisingData('/home/dircon/yuna/AIHub', 'train')
    print(len(dataset))
    
    i, g = dataset[0]
    print(i.shape)
    print(g.shape)