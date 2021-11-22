import os
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import imgaug.augmenters as iaa
import random
from matplotlib import pyplot as plt


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, scale=1, color_map='RGB', split='train'):
        self.imgs_dir = imgs_dir
        self.scale = scale
        self.split = split
        self.color_map = color_map
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        random.seed(1)

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


    def augmentation(self, pil_img):
        input_img = np.expand_dims(pil_img, axis=0)

        prob = random.uniform(0, 1)
        if self.split == 'train' and prob > 0.5:  # we do augmentation in 50% of the cases
            seq = iaa.Sequential([
                iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace=self.color_map),
                iaa.ChannelShuffle(0.35),

                iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
                iaa.Affine(rotate=(-180, 180)),
                iaa.Affine(shear=(-16, 16)),
                iaa.Fliplr(0.5),
                iaa.GaussianBlur(sigma=(0, 3.0))
            ])
            images_aug = seq(images=input_img)

            # if we would like to see the data augmentation
            # segmaps_aug = np.concatenate((segmaps_aug,segmaps_aug,segmaps_aug), 3)
            # seq.show_grid([images_aug[0], segmaps_aug[0]*255], cols=16, rows=8)

            output_img = np.transpose(images_aug[0], (2, 0, 1))
        else:
            seq = iaa.Sequential([
                iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace=self.color_map),
                iaa.Pad(px=64,pad_mode='edge', keep_size=False),
                iaa.Resize(512)
            ])
            images_aug = seq(images=input_img)

            output_img = np.transpose(images_aug[0], (2, 0, 1))

        return output_img

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = glob(os.path.join(self.imgs_dir, idx +'.png'))
        img = Image.open(img_file[0])
        # img = img.resize((512, 512))

        img = np.array(img)

        img = self.augmentation(img)
        # img = img.swapaxes(0, 1)
        # img = img.swapaxes(1, 2)
        # plt.imshow(img)
        # plt.show()
        if img.max() > 1:
            img = img / 255
        mask = []
        return {'image': torch.from_numpy(img), 'filename': idx}