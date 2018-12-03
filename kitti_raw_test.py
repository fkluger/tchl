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

# Change this to the directory where you store KITTI data
basedir = '/tnt/data/scene_understanding/KITTI/rawdata'

dates = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']

for date in dates:

    date_dir = basedir + "/" + date

    drive_dirs = glob.glob(date_dir + "/*sync")
    drive_dirs.sort()

    drives = []
    for drive_dir in drive_dirs:
        drive = drive_dir.split("_")[-2]
        drives.append(drive)


    # Specify the dataset to load
    for drive in drives:
    # drive = '0001'

        print(date, drive)

        dataset = pykitti.raw(basedir, date, drive)

        target_dir = "/home/kluger/tmp/kitti_horizons_raw/%s/%s" % (date, drive)

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        M = np.matrix(np.eye(4))

        H = np.matrix([[0.], [1.], [0.]])
        print("H: ", H)

        calib0 = dataset.calib.P_rect_20
        P = np.matrix(calib0)
        R_cam_imu = np.matrix(dataset.calib.T_cam2_imu[0:3,0:3])
        K = np.matrix(dataset.calib.P_rect_20[0:3, 0:3])

        for idx, gray in enumerate(iter(dataset.rgb)):

            R_imu = np.matrix(dataset.oxts[idx].T_w_imu[0:3,0:3])

            print(idx)

            G = np.matrix([[0.], [0.], [1.]])
            G_imu = R_imu.T * G
            G_cam = R_cam_imu * G_imu

            # print(G)
            # print(G_imu)
            # print(G_cam)

            image = gray[0]

            width = image.width
            height = image.height

            h = np.array(K.I.T*G_cam).squeeze()

            rotation = 5
            shift = (-20., 20., 0)

            rot = -rotation/180.*np.pi
            Tf = np.matrix([[1, 0, -width/2.], [0, 1, -height/2.], [0, 0, 1]])
            Tb = np.matrix([[1, 0, width/2.], [0, 1, height/2.], [0, 0, 1]])
            Rt = Tb*np.matrix([[np.cos(rot), -np.sin(rot), -shift[0]], [np.sin(rot), np.cos(rot), -shift[1]], [0,0,1]])*Tf

            h = np.array(Rt.I.T * np.matrix(h).T).squeeze()

            hp1 = np.cross(h, np.array([1, 0, 0]))
            hp2 = np.cross(h, np.array([1, 0, -gray[0].width]))

            hp1 /= hp1[2]
            hp2 /= hp2[2]





            image = ndimage.interpolation.rotate(image, rotation, reshape=False, mode='nearest')
            image = ndimage.interpolation.shift(image, shift, mode='nearest')

            f, ax = plt.subplots(1, 1, figsize=(15, 5))
            ax.imshow(image)
            # ax.set_title('Left Gray Image (cam0)')
            plt.autoscale(False)
            ax.plot([hp1[0], hp2[0]], [hp1[1], hp2[1]], 'g-', lw=2)

            plt.axis('off')

            # plt.savefig("%s/%06d.png" % (target_dir, idx), bbox_inches='tight')
            # f2 = plt.figure()
            # ax2 = f2.add_subplot(111, projection='3d')
            # # Plot every 100th point so things don't get too bogged down
            # velo_range = range(0, third_velo.shape[0], 100)
            # ax2.scatter(third_velo[velo_range, 0],
            #             third_velo[velo_range, 1],
            #             third_velo[velo_range, 2],
            #             c=third_velo[velo_range, 3],
            #             cmap='gray')
            # ax2.set_title('Third Velodyne scan (subsampled)')

            plt.show()
            # plt.close()

            # pickle_file = "%s/%06d.pkl" % (target_dir, idx)

            # with open(pickle_file, 'wb') as f:
            #     pickle.dump((h, G_cam, hp1, hp2), f, -1)