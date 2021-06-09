from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
import os
import torch.nn.functional as F
from torch.utils.data import Dataset
import logging
from PIL import Image
import torchvision.transforms as transforms
import kornia as K
import random

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.transform = transforms.Compose([
                  transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0),
                  transforms.RandomHorizontalFlip(p=0.3),
                  transforms.RandomVerticalFlip(p=0.3),
                  transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                  ])

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file1 = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        mask_file = str(mask_file1)[2:-2]
        img_file = glob(self.imgs_dir + idx + '.*')
        img1 = Image.open(img_file[0])
        mask1= torch.load(mask_file)
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.transform is not None:
            img = self.transform(img1)
        random.seed(seed) # apply this seed to target tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.transform is not None:
            mask = self.transform(mask1)
        return {
            'image': transforms.ToTensor()(img),
            'mask': mask
        }

class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
