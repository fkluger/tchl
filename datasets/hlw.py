from torch.utils.data import Dataset
import csv
import pykitti
import numpy as np
import random
import time
import pickle
from scipy import ndimage
import glob
import torch
from torchvision import transforms
import cv2
from PIL import Image
import os

WIDTH = 512
HEIGHT = 512


class HLWDataset(Dataset):

    def __init__(self, root_dir, set='train', augmentation=True, transform=None):

        self.root_dir = root_dir
        self.augmentation = augmentation
        self.transform = transform

        split_file = os.path.join(root_dir, "split/%s.txt" % set)

        with open(split_file, 'r') as f:
            file_list = f.readlines()
        self.file_list = [x.strip() for x in file_list]

        csv_path = os.path.join(root_dir, "metadata.csv")
        self.annotations = {}

        with open(csv_path) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for row in reader:
                self.annotations[row[0]] = [float(x) for x in row[1:]]

    def __len__(self):
        return len(self.file_list)

    def image_path(self, image_name):
        return os.path.join(self.root_dir, "images/" + image_name)

    def __getitem__(self, idx):

        images = np.zeros((1, 3, HEIGHT, WIDTH)).astype(np.float32)
        offsets = np.zeros((1, 1)).astype(np.float32)
        angles = np.zeros((1, 1)).astype(np.float32)

        image_id = self.file_list[idx]
        path = self.image_path(image_id)

        image = Image.open(path)
        annot = self.annotations[image_id]
        horizon_coords = np.array([annot[2:6]]).squeeze()

        if self.augmentation:
            scale_w = WIDTH * 1. / image.width
            scale_h = HEIGHT * 1. / image.height
            scale_lo = np.minimum(scale_h, scale_w)
            scale_hi = np.maximum(scale_h, scale_w)
            scale = np.random.uniform(scale_lo, scale_hi)
        else:
            scale = WIDTH * 1. / np.minimum(image.width, image.height)

        new_width = int(np.around(image.width * scale))
        new_height = int(np.around(image.height * scale))

        scale_w = new_width * 1. / image.width
        scale_h = new_height * 1. / image.height

        horizon_coords[0] *= scale_w
        horizon_coords[2] *= scale_w
        horizon_coords[1] *= scale_h
        horizon_coords[3] *= scale_h

        image = image.resize((new_width, new_height), Image.BICUBIC)

        crop_margin_w = new_width - WIDTH
        crop_margin_h = new_height - HEIGHT

        left = 0
        right = new_width - 1
        upper = 0
        lower = new_height - 1

        if crop_margin_w > 0:
            left = np.random.randint(0, crop_margin_w) if self.augmentation else crop_margin_w/2.
            right = left + WIDTH
        if crop_margin_h > 0:
            upper = np.random.randint(0, crop_margin_h) if self.augmentation else crop_margin_h/2.
            lower = upper + HEIGHT

        horizon_coords[0] -= left/2. - (new_width-right)/2.
        horizon_coords[2] -= left/2. - (new_width-right)/2.
        horizon_coords[1] -= upper/2. - (new_height-lower)/2.
        horizon_coords[3] -= upper/2. - (new_height-lower)/2.

        image = image.crop((left, upper, right, lower))

        pad_w = WIDTH - image.width
        pad_h = HEIGHT - image.height

        if pad_w > 0:
            pad_w1 = np.random.randint(0, int(pad_w)) if self.augmentation else int(pad_w/2)
            pad_w2 = pad_w - pad_w1
        else:
            pad_w1 = 0
            pad_w2 = 0
        if pad_h > 0:
            pad_h1 = np.random.randint(0, int(pad_h)) if self.augmentation else int(pad_h/2)
            pad_h2 = pad_h - pad_h1
        else:
            pad_h1 = 0
            pad_h2 = 0

        padded_image = np.pad(np.array(image), ((pad_h1, pad_h2), (pad_w1, pad_w2), (0, 0)), 'edge')

        horizon_coords[0] += pad_w1/2. - pad_w2/2.
        horizon_coords[2] += pad_w1/2. - pad_w2/2.
        horizon_coords[1] -= pad_h1/2. - pad_h2/2.
        horizon_coords[3] -= pad_h1/2. - pad_h2/2.

        hp1 = np.array([horizon_coords[0], horizon_coords[1], 1.])
        hp2 = np.array([horizon_coords[2], horizon_coords[3], 1.])
        h = np.cross(hp1, hp2)
        hp1 = np.cross(h, np.array([1, 0, WIDTH/2]))
        hp2 = np.cross(h, np.array([1, 0, -WIDTH/2]))
        hp1 /= hp1[2]
        hp2 /= hp2[2]



        mh = (0.5 * (hp1[1] + hp2[1])) / HEIGHT
        offset = -mh

        angle = np.arctan2(h[0], h[1])
        if angle > np.pi / 2:
            angle -= np.pi
        elif angle < -np.pi / 2:
            angle += np.pi

        # print(offset, angle)

        if self.transform is not None:
            image = self.transform(Image.fromarray((padded_image).astype('uint8')))
        else:
            image = np.transpose(padded_image, [2, 0, 1])

        images[0,:,:,:] = image
        # sample = {'images': images, 'offsets': offsets, 'angles': angles}

        offsets[0,0] = offset
        angles[0,0] = angle

        sample = {'images': images, 'offsets': offsets, 'angles': angles}

        return sample




class Cutout(object):
    def __init__(self, length, bias=False):
        self.length = length
        self.central_bias = bias

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        if self.central_bias:
            x = int(np.around(w/4. * np.random.rand(1) + w/2.))
            y = int(np.around(h/4. * np.random.rand(1) + h/2.))

        else:
            y = np.random.randint(h)
            x = np.random.randint(w)

        lx = np.random.randint(1, self.length)
        ly = np.random.randint(1, self.length)

        y1 = np.clip(y - ly // 2, 0, h)
        y2 = np.clip(y + ly // 2, 0, h)
        x1 = np.clip(x - lx // 2, 0, w)
        x2 = np.clip(x + lx // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    tfs = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = HLWDataset("/tnt/data/scene_understanding/HLW/", set='test', augmentation=False, transform=tfs)

    print("dataset size: ", len(dataset))

    for idx, sample in enumerate(dataset):
        images = sample['images']
        offsets = sample['offsets']
        angles = sample['angles']
        image = images[0, :, :, :].transpose((1, 2, 0))
        width = image.shape[1]
        height = image.shape[0]

        offset = offsets.squeeze()
        offset += 0.5
        offset *= height
        angle = angles.squeeze()

        true_mp = np.array([width / 2., offset])
        true_nv = np.array([np.sin(angle), np.cos(angle)])
        true_hl = np.array([true_nv[0], true_nv[1], -np.dot(true_nv, true_mp)])
        true_h1 = np.cross(true_hl, np.array([1, 0, 0]))
        true_h2 = np.cross(true_hl, np.array([1, 0, -width]))
        true_h1 /= true_h1[2]
        true_h2 /= true_h2[2]

        plt.figure()
        plt.imshow(image)
        plt.autoscale(False)
        plt.plot([true_h1[0], true_h2[0]], [true_h1[1], true_h2[1]], 'g-', lw=4)
        plt.show()

