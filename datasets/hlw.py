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

    def __init__(self, root_dir, set='train', augmentation=True, transform=None, scale=1., get_images=True):

        self.root_dir = root_dir
        self.augmentation = augmentation
        self.transform = transform
        self.get_images = get_images

        self.height = int(HEIGHT*scale)
        self.width = int(WIDTH*scale)

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

        images = np.zeros((1, 3, self.height, self.width)).astype(np.float32)
        offsets = np.zeros((1, 1)).astype(np.float32)
        angles = np.zeros((1, 1)).astype(np.float32)

        image_id = self.file_list[idx]
        path = self.image_path(image_id)

        image = Image.open(path)
        annot = self.annotations[image_id]
        horizon_coords = np.array([annot[2:6]]).squeeze()

        if self.augmentation:
            rotation = np.random.uniform(-2, 2)
            max_shift = 0.

            shift = (0,0,0)#(np.random.uniform(-max_shift, max_shift), np.random.uniform(-max_shift, max_shift), 0)

            rot = -rotation / 180. * np.pi
            Tf = np.matrix([[1, 0, -self.width / 2.], [0, 1, -self.height / 2.], [0, 0, 1]])
            Tb = np.matrix([[1, 0, self.width / 2.], [0, 1, self.height / 2.], [0, 0, 1]])
            Rt = Tb * np.matrix(
                [[np.cos(rot), -np.sin(rot), -shift[0]], [np.sin(rot), np.cos(rot), -shift[1]], [0, 0, 1]]) * Tf

        if self.augmentation:
            scale_w = self.width * 1. / image.width
            scale_h = self.height * 1. / image.height
            scale_lo = np.minimum(scale_h, scale_w)
            scale_hi = np.maximum(scale_h, scale_w)
            scale = np.random.uniform(scale_lo, scale_hi)
        else:
            # scale = self.width * 1. / np.minimum(image.width, image.height)
            scale = self.width * 1./image.height

        new_width = int(np.around(image.width * scale))
        new_height = int(np.around(image.height * scale))
        # print(new_height, new_width)

        horizon_coords[1] *= -1
        horizon_coords[3] *= -1

        horizon_coords += np.array([image.width/2., image.height/2., image.width/2., image.height/2.])

        scale_w = new_width * 1. / image.width
        scale_h = new_height * 1. / image.height

        horizon_coords[0] *= scale_w
        horizon_coords[2] *= scale_w
        horizon_coords[1] *= scale_h
        horizon_coords[3] *= scale_h

        image = image.resize((new_width, new_height), Image.BICUBIC)

        crop_margin_w = new_width - self.width
        crop_margin_h = new_height - self.height

        left = 0
        right = new_width
        upper = 0
        lower = new_height

        if crop_margin_w > 0:
            # left = crop_margin_w/2.
            left = np.random.randint(0, crop_margin_w) if self.augmentation else crop_margin_w/2.
            right = left + self.width
        if crop_margin_h > 0:
            # upper = crop_margin_h/2.
            upper = np.random.randint(0, crop_margin_h) if self.augmentation else crop_margin_h/2.
            lower = upper + self.height

        horizon_coords[0] -= left#/2. - (new_width-right)/2.
        horizon_coords[2] -= left#/2. - (new_width-right)/2.
        horizon_coords[1] -= upper#/2. - (new_height-lower)/2.
        horizon_coords[3] -= upper#/2. - (new_height-lower)/2.

        image = image.crop((left, upper, right, lower))

        pad_w = self.width - image.width
        pad_h = self.height - image.height
        # print("pad: ", pad_w, pad_h)

        if pad_w > 0:
            # pad_w1 = int(pad_w/2)
            pad_w1 = np.random.randint(0, int(pad_w)) if self.augmentation else int(pad_w/2)
            pad_w2 = pad_w - pad_w1
        else:
            pad_w1 = 0
            pad_w2 = 0
        if pad_h > 0:
            # pad_h1 = int(pad_h/2)
            pad_h1 = np.random.randint(0, int(pad_h)) if self.augmentation else int(pad_h/2)
            pad_h2 = pad_h - pad_h1
        else:
            pad_h1 = 0
            pad_h2 = 0

        padded_image = np.pad(np.array(image), ((pad_h1, pad_h2), (pad_w1, pad_w2), (0, 0)), 'edge')

        horizon_coords[0] += pad_w1 #/2. - pad_w2/2.
        horizon_coords[2] += pad_w1 #/2. - pad_w2/2.
        horizon_coords[1] += pad_h1 #/2. - pad_h2/2.
        horizon_coords[3] += pad_h1 #/2. - pad_h2/2.

        hp1 = np.array([horizon_coords[0], horizon_coords[1], 1.])
        hp2 = np.array([horizon_coords[2], horizon_coords[3], 1.])
        # h = np.cross(hp1, hp2)
        # hp1 = np.cross(h, np.array([1, 0, self.width/2]))
        # hp2 = np.cross(h, np.array([1, 0, -self.width/2]))
        # hp1 /= hp1[2]
        # hp2 /= hp2[2]

        # hp1 = np.array([hp1[0]+self.width/2, -hp1[1]+self.height/2, 1.])
        # hp2 = np.array([hp2[0]+self.width/2, -hp2[1]+self.height/2, 1.])

        h = np.cross(hp1, hp2)

        hp1 = np.cross(h, np.array([1, 0, 0]))
        hp2 = np.cross(h, np.array([1, 0, -self.width]))
        hp1 /= hp1[2]
        hp2 /= hp2[2]


        # print(offset, angle)

        if self.augmentation:

            h = np.array(Rt.I.T * np.matrix(h).T).squeeze()

            angle = np.arctan2(h[0], h[1])
            if angle > np.pi / 2:
                angle -= np.pi
            elif angle < -np.pi / 2:
                angle += np.pi

            M = cv2.getRotationMatrix2D((padded_image.shape[1] / 2, padded_image.shape[0] / 2), rotation, 1)
            M[0, 2] += shift[0]
            M[1, 2] += -shift[1]
            padded_image = cv2.warpAffine(padded_image, M, (0, 0), borderMode=cv2.BORDER_REPLICATE)

            if self.augmentation and np.random.uniform(0., 1.) > 0.5:
                padded_image = cv2.flip(padded_image, 1)
                angle *= -1

            hp1 = np.cross(h, np.array([1, 0, 0]))
            hp2 = np.cross(h, np.array([1, 0, -self.width]))
            hp1 /= hp1[2]
            hp2 /= hp2[2]

        else:
            angle = np.arctan2(h[0], h[1])
            if angle > np.pi / 2:
                angle -= np.pi
            elif angle < -np.pi / 2:
                angle += np.pi

        offset = (0.5 * (hp1[1] + hp2[1])) / self.height - 0.5

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

    dataset1 = HLWDataset("/data/scene_understanding/HLW/", set='train', augmentation=True, transform=tfs)
    # dataset2 = HLWDataset("/data/scene_understanding/HLW/", set='train', augmentation=True, transform=tfs)

    # print("dataset size: ", len(dataset))

    all_offsets = []
    all_angles = []

    for idx, sample in enumerate(dataset1):
        # if idx > 50: break
        print(idx+1, " / ", len(dataset1))
        images = sample['images']
        offsets = sample['offsets']
        angles = sample['angles']
        image = images[0, :, :, :].transpose((1, 2, 0))
        width = image.shape[1]
        height = image.shape[0]

        offset = offsets.squeeze()
        angle = angles.squeeze()
        all_angles += [angle]
        all_offsets += [offset]
        offset += 0.5
        offset *= height
        print("angle: ", angle*180/np.pi)

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
        plt.plot([true_h1[0], true_h2[0]], [true_h1[1], true_h2[1]], 'g-', lw=2)
        plt.show()

        exit(0)

    angle_mean = np.mean(all_angles)
    angle_std = np.std(all_angles)
    angle_min = np.min(all_angles)
    angle_max = np.max(all_angles)

    print("mean: ", angle_mean)
    print("stdd: ", angle_std)
    print("min: ", angle_min)
    print("max: ", angle_max)

    plt.figure()
    n, bins, patches = plt.hist(all_angles, 100, density=True, facecolor='g', alpha=0.75)
    plt.show()


        # images = sample2['images']
        # offsets = sample2['offsets']
        # angles = sample2['angles']
        # image = images[0, :, :, :].transpose((1, 2, 0))
        # width = image.shape[1]
        # height = image.shape[0]
        #
        # offset = offsets.squeeze()
        # offset += 0.5
        # offset *= height
        # angle = angles.squeeze()
        #
        # true_mp = np.array([width / 2., offset])
        # true_nv = np.array([np.sin(angle), np.cos(angle)])
        # true_hl = np.array([true_nv[0], true_nv[1], -np.dot(true_nv, true_mp)])
        # true_h1 = np.cross(true_hl, np.array([1, 0, 0]))
        # true_h2 = np.cross(true_hl, np.array([1, 0, -width]))
        # true_h1 /= true_h1[2]
        # true_h2 /= true_h2[2]
        #
        # plt.figure()
        # plt.imshow(image)
        # plt.autoscale(False)
        # plt.plot([true_h1[0], true_h2[0]], [true_h1[1], true_h2[1]], 'g-', lw=4)
        # plt.show()

        # exit(0)