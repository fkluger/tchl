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
from scipy import ndimage, io
from skimage import transform

# Change this to the directory where you store KITTI data
# basedir = '/phys/intern/kluger/tmp/kitti/rawdata'
basedir = '/data/scene_understanding/KITTI/rawdata'

dates = [
    '2011_09_26',
    '2011_09_28',
    '2011_09_29',
    '2011_09_30',
    '2011_10_03'
]

# WIDTH = 1250
# HEIGHT = 380

scale = 1.

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

        # target_dir = "/phys/intern/kluger/tmp/kitti/mat_horizons_s%.3f%s/%s" % (scale, date, drive)
        target_dir = "/data/kluger/datasets/kitti/mat_horizons_s%.3f/%s/%s" % (scale, date, drive)

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        R_cam_imu = np.matrix(dataset.calib.T_cam2_imu[0:3,0:3])
        K = np.matrix(dataset.calib.P_rect_20[0:3, 0:3])

        G = np.matrix([[0.], [0.], [1.]])

        offsets = []
        angles = []

        offset_ema = 0
        angle_ema = 0

        for idx, image in enumerate(iter(dataset.rgb)):

            image = np.array(image[0])
            orig_image_width = image.shape[1]
            orig_image_height = image.shape[0]

            if scale < 1.:
                image = transform.rescale(image, scale)

            image_width = image.shape[1]
            image_height = image.shape[0]
            # print(image.shape)

            # image_width = WIDTH

            # pad_w = WIDTH - image[0].width
            # pad_h = HEIGHT - image[0].height
            #
            # pad_w1 = int(pad_w / 2)
            # pad_w2 = pad_w - pad_w1
            # pad_h1 = int(pad_h / 2)
            # pad_h2 = pad_h - pad_h1
            #
            # padded_image = np.pad(np.array(image[0]), ((pad_h1, pad_h2), (pad_w1, pad_w2), (0, 0)), 'edge')
            # if scale < 1.:
            #     padded_image = transform.rescale(padded_image, scale)

            R_imu = np.matrix(dataset.oxts[idx].T_w_imu[0:3, 0:3])
            G_imu = R_imu.T * G
            G_cam = R_cam_imu * G_imu

            h = np.array(K.I.T * G_cam).squeeze()

            # padded_image = np.transpose(padded_image, [2, 0, 1]).astype(np.float32) #/ 255.

            hp1 = np.cross(h, np.array([1, 0, 0]))
            hp2 = np.cross(h, np.array([1, 0, -orig_image_width]))
            hp1 /= hp1[2]
            hp2 /= hp2[2]

            hp1[0:2] *= scale
            hp2[0:2] *= scale

            # hp1[1] *= -1
            # hp2[1] *= -1

            # hp1[1] += image_height
            # hp2[1] += image_height

            y1_gt = hp1[1]
            y2_gt = hp2[1]

            # hp1[0] += pad_w1
            # hp2[0] += pad_w1
            # hp1[1] += pad_h1
            # hp2[1] += pad_h1


            # offset = (0.5 * (hp1[1] + hp2[1])) / image_height - 0.5
            # mp_old = 0.5 * (hp1 + hp2)
            #
            # h = np.cross(hp1, hp2)
            #
            # angle = np.arctan2(h[0], h[1])
            # if angle > np.pi / 2:
            #     angle -= np.pi
            # elif angle < -np.pi / 2:
            #     angle += np.pi
            #
            # if idx == 0:
            #     offset_ema = offset
            #     angle_ema = angle
            # else:
            #     offset_ema = offset * ema_alpha + (1-ema_alpha) * offset_ema
            #     angle_ema = angle * ema_alpha + (1-ema_alpha) * angle_ema
            #
            # offset_ema_ = 0.5 + offset_ema
            # offset_ema_ *= HEIGHT

            # mp = np.array([WIDTH / 2. * scale, offset_ema_])
            # nv = np.array([np.sin(angle_ema), np.cos(angle_ema)])
            # h_ema = np.array([nv[0], nv[1], -np.dot(nv, mp)])
            # h = h/np.linalg.norm(h[0:2])

            # data = {'image': padded_image, 'horizon_hom': h, 'horizon_p1': hp1, 'horizon_p2': hp2, 'offset': offset,
            #         'angle': angle, 'offset_ema': offset_ema, 'angle_ema': angle_ema, 'horizon_hom_ema': h_ema,
            #         'scale': scale, 'padding': (pad_w1, pad_w2, pad_h1, pad_h2), 'G': G_cam, 'K': K}

            mat_file = target_dir + "/%06d.mat" % idx

            io.savemat(mat_file, {'image': image, 'img': image, 'hp1': hp1, 'hp2': hp2, 'y1_gt': y1_gt, 'y2_gt': y2_gt,
                                  'G': G_cam.squeeze(), 'K': K})

            # with open(pickle_file, 'wb') as f:
            #     pickle.dump(data, f, -1)

            # offsets += [offset]
            # angles += [angle]

        # indices = np.array(list(range(len(offsets))))
        # offsets = np.array(offsets)
        # angles = np.array(angles)
        #
        # alpha = 0.1
        # offsets_smooth = offsets.copy()
        # for j in range(1,offsets_smooth.shape[0]):
        #     offsets_smooth[j] = alpha * offsets_smooth[j] + (1-alpha) * offsets_smooth[j-1]
        #
        # alpha = 0.1
        # angles_smooth = angles.copy()
        # for j in range(1,angles_smooth.shape[0]):
        #     angles_smooth[j] = alpha * angles_smooth[j] + (1-alpha) * angles_smooth[j-1]
        #
        # diff_o = offsets - offsets_smooth
        # diff_a = angles - angles_smooth
        #
        # # plt.plot(indices, offsets, 'g-')
        # # plt.plot(indices, offsets_smooth, 'c-')
        # # plt.plot(indices, diff_o, 'b-')
        # plt.plot(indices, angles, 'g-')
        # plt.plot(indices, angles_smooth, 'c-')
        # plt.plot(indices, diff_a, 'b-')
        # # plt.ylim(-.3, .3)
        # plt.show()
