import pickle
import matplotlib.pyplot as plt
import numpy as np
import platform
import glob
import os
import csv
import time

from datasets.kitti import KittiRawDatasetPP, Cutout, HEIGHT, WIDTH
from torchvision import transforms
import torch

hostname = platform.node()

if 'daidalos' in hostname:
    target_base = "/tnt/data/kluger/checkpoints/horizon_sequences"
    root_dir = "/tnt/data/kluger/datasets/kitti/horizons"
    csv_base = "/tnt/home/kluger/tmp/kitti_split_3"
    pdf_file = "/tnt/home/kluger/tmp/kitti_split/data_pdfs.pkl"
elif 'athene' in hostname:
    target_base = "/data/kluger/checkpoints/horizon_sequences"
    root_dir = "/phys/intern/kluger/tmp/kitti/horizons"
    csv_base = "/home/kluger/tmp/kitti_split_3"
    pdf_file = "/home/kluger/tmp/kitti_split/data_pdfs.pkl"
else:
    target_base = "/data/kluger/checkpoints/horizon_sequences"
    root_dir = "/data/kluger/datasets/kitti/horizons"
    csv_base = "/home/kluger/tmp/kitti_split_3"
    pdf_file = "/home/kluger/tmp/kitti_split/data_pdfs.pkl"

downscale = 2
sequence_length = 256

if downscale > 1:
    root_dir += "_s%.3f" % (1. / downscale)

root_dir += "_ema0.100"

pixel_mean = [0.362365, 0.377767, 0.366744]

tfs = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomGrayscale(p=0.1),
    # transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    #transforms.Normalize(mean=pixel_mean, std=[1., 1., 1.]),
    Cutout(length=500//downscale)
])
tfs_val = transforms.Compose([
    transforms.ToTensor(),
   #  transforms.Normalize(mean=pixel_mean, std=[1., 1., 1.]),
])


# tfs=None
train_dataset = KittiRawDatasetPP(root_dir=root_dir, pdf_file=None, fill_up=False, return_info=True,
                                  csv_file=csv_base + "/train.csv", seq_length=sequence_length,
                                  im_height=HEIGHT // downscale, im_width=WIDTH // downscale,
                                  scale=1. / downscale, transform=tfs)
val_dataset = KittiRawDatasetPP(root_dir=root_dir, pdf_file=None, augmentation=False, fill_up=False,
                                csv_file=csv_base + "/val.csv", seq_length=sequence_length, return_info=True,
                                im_height=HEIGHT // downscale, im_width=WIDTH // downscale,
                                scale=1. / downscale, transform=tfs_val)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=1,
                                           shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=1,
                                           shuffle=False)


if __name__ == '__main__':
    for idx, sample in enumerate(val_loader):

        images = sample['images']
        offsets = sample['offsets']
        angles = sample['angles']
        Gs = sample['G']

        for si in range(images.shape[1]):

            image = images.numpy()[0, si, :, :, :].transpose((1, 2, 0))
            width = image.shape[1]
            height = image.shape[0]

            offset = offsets[0, si].numpy().squeeze()
            angle = angles[0, si].numpy().squeeze()
            offset += 0.5
            offset *= height

            true_mp = np.array([width / 2., offset])
            true_nv = np.array([np.sin(angle), np.cos(angle)])
            true_hl = np.array([true_nv[0], true_nv[1], -np.dot(true_nv, true_mp)])
            true_h1 = np.cross(true_hl, np.array([1, 0, 0]))
            true_h2 = np.cross(true_hl, np.array([1, 0, -width]))
            true_h1 /= true_h1[2]
            true_h2 /= true_h2[2]

            plt.figure()
            plt.imshow(image)

            plt.plot([true_h1[0], true_h2[0]], [true_h1[1], true_h2[1]], '-', lw=4, c='#99C000')

            plt.show()
            time.sleep(0.5)
