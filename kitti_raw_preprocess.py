"""Example of pykitti.odometry usage."""
import itertools
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import pykitti
import pickle
import glob
from scipy import ndimage
from skimage import transform

# Change this to the directory where you store KITTI data
basedir = '/phys/intern/kluger/tmp/kitti/rawdata'
# basedir = '/home/kluger/athene/kluger/tmp/kitti/rawdata'

dates = [
    '2011_09_26',
    '2011_09_28',
    '2011_09_29',
    '2011_09_30',
    '2011_10_03'
]

WIDTH = 1250
HEIGHT = 380

scale = 0.5

for date in dates:

    date_dir = basedir + "/" + date

    drive_dirs = glob.glob(date_dir + "/*sync")
    drive_dirs.sort()

    drives = []
    for drive_dir in drive_dirs:
        drive = drive_dir.split("_")[-2]
        drives.append(drive)

    for drive in drives:

        print(date, drive)

        dataset = pykitti.raw(basedir, date, drive)

        target_dir = "/phys/intern/kluger/tmp/kitti/horizons_s%.3f/%s/%s" % (scale, date, drive)
        # target_dir = "/home/kluger/athene/kluger/tmp/kitti/horizons_s%.3f/%s/%s" % (scale, date, drive)

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        R_cam_imu = np.matrix(dataset.calib.T_cam2_imu[0:3,0:3])
        K = np.matrix(dataset.calib.P_rect_20[0:3, 0:3])

        G = np.matrix([[0.], [0.], [1.]])

        for idx, image in enumerate(iter(dataset.rgb)):

            image_width = WIDTH

            pad_w = WIDTH - image[0].width
            pad_h = HEIGHT - image[0].height

            pad_w1 = int(pad_w / 2)
            pad_w2 = pad_w - pad_w1
            pad_h1 = int(pad_h / 2)
            pad_h2 = pad_h - pad_h1

            padded_image = np.pad(np.array(image[0]), ((pad_h1, pad_h2), (pad_w1, pad_w2), (0, 0)), 'edge')
            padded_image = transform.rescale(padded_image, scale)

            R_imu = np.matrix(dataset.oxts[idx].T_w_imu[0:3, 0:3])
            G_imu = R_imu.T * G
            G_cam = R_cam_imu * G_imu

            h = np.array(K.I.T * G_cam).squeeze()

            padded_image = np.transpose(padded_image, [2, 0, 1]).astype(np.float32) #/ 255.

            hp1 = np.cross(h, np.array([1, 0, 0]))
            hp2 = np.cross(h, np.array([1, 0, -image_width]))
            hp1 /= hp1[2]
            hp2 /= hp2[2]

            hp1[0] += pad_w1
            hp2[0] += pad_w1
            hp1[1] += pad_h1
            hp2[1] += pad_h1

            hp1[0:2] *= scale
            hp2[0:2] *= scale

            offset = (0.5 * (hp1[1] + hp2[1])) / HEIGHT - 0.5

            h = np.cross(hp1, hp2)

            angle = np.arctan2(h[0], h[1])
            if angle > np.pi / 2:
                angle -= np.pi
            elif angle < -np.pi / 2:
                angle += np.pi

            data = {'image': padded_image, 'horizon_hom': h, 'horizon_p1': hp1, 'horizon_p2': hp2, 'offset': offset,
                    'angle': angle}

            pickle_file = target_dir + "/%06d.pkl" % idx

            with open(pickle_file, 'wb') as f:
                pickle.dump(data, f, -1)
