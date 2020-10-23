"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/nv-tlabs/meta-sim.
Authors: Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci, Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba and Sanja Fidler
"""

import glob
import os.path as osp

import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


class CarlaLoader(data.Dataset):
    def __init__(self, root):
        super(CarlaLoader, self).__init__()
        self.root = root
        self.files = glob.glob(osp.join(self.root, '*.jpg'))
        self.files = [''.join(f.split('.')[:-1]) for f in self.files]

    def __getitem__(self, index):
        print(f'CarlaLoader.__getitem__({index})')
        img = self.pull_image(index)

        img = img.transpose((2, 0, 1))  # convert to HWC
        img = (torch.FloatTensor(img) / 255.0)
        return img, None

    def pull_image(self, index):
        image = Image.open(self.files[index] + '.jpg').convert('RGB')
        return np.asarray(image)

    def __len__(self):
        return len(self.files)

    def get_bunch_images(self, num):
        assert (num < len(self), 'Asked for more images than size of data')

        idxs = np.random.choice(range(len(self)), num, replace=False)

        return np.array([self.pull_image(idx) for idx in idxs], dtype=np.float32)
