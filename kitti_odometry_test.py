"""Example of pykitti.odometry usage."""
import itertools
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import pykitti
import pickle

# Change this to the directory where you store KITTI data
basedir = '/tnt/data/scene_understanding/KITTI/odometry'

# Specify the dataset to load
sequence = '00'

# Load the data. Optionally, specify the frame range to load.
# dataset = pykitti.odometry(basedir, sequence)
dataset = pykitti.odometry(basedir, sequence)


def vp_from_lines(lines, use_weight=False, im_centre=None):

    if use_weight:
        for li in range(lines.shape[0]):
            l = lines[li,:].copy()
            l /= l[2]
            l[0] -= im_centre[0]
            l[1] -= im_centre[1]
            w = 1/np.maximum(np.linalg.norm(l[0:2]), 1e-6)
            lines[li,:] *= w


    Mat = np.matrix(lines)

    U, S, V = np.linalg.svd(Mat.T * Mat)

    V = np.array(V.T)

    vp = np.squeeze(V[:, 2])

    vp /= vp[2]

    # vp *= np.sign(vp[2])

    return vp


class EventHandler:
    def __init__(self, fig, ax, im_centre, K):
        self.p1 = None
        self.p2 = None

        self.fig = fig
        self.ax = ax

        self.lines = []
        self.vps = []

        self.horizon = None

        self.im_centre = im_centre

        self.K = K
        self.n = None


    def onclick(self, event):
        x = event.xdata
        y = event.ydata

        if self.p1 is None:
            self.p1 = np.array([x, y, 1.])
        elif self.p2 is None:
            self.p2 = np.array([x, y, 1.])
            self.ax.plot([self.p1[0], self.p2[0]], [self.p1[1], self.p2[1]], 'g-')
            self.fig.canvas.draw()
            line = np.cross(self.p1, self.p2)
            line /= np.linalg.norm(line)
            self.lines.append(line)

            self.p1 = None
            self.p2 = None
        else:
            assert False

    def onpress(self, event):
        if event.key == 'n':
            if len(self.lines) > 1:
                vp = vp_from_lines(self.lines)
                vp /= np.linalg.norm(vp)
                print(vp)

                self.n = K.I * np.matrix(vp).T
                self.n /= np.linalg.norm(self.n)

                self.horizon = K.I.T * self.n

                self.vps.append(vp)
                self.lines.clear()

                h = np.array(self.horizon).squeeze()
                hp1 = np.cross(h, np.array([1, 0, 0]))
                hp2 = np.cross(h, np.array([1, 0, -gray[0].width]))

                hp1 /= hp1[2]
                hp2 /= hp2[2]
                self.ax.plot([hp1[0], hp2[0]], [hp1[1], hp2[1]], 'm-', lw=2)
                self.fig.canvas.draw()

        elif event.key == 'z':
            self.lines.clear()

        else:
            # if len(self.vps) > 0:
            #     newVPs = np.vstack(self.vps)
            #     self.horizon = vp_from_lines(newVPs, use_weight=True, im_centre=self.im_centre)

            plt.close()


# dataset.calib:      Calibration data are accessible as a named tuple
# dataset.timestamps: Timestamps are parsed into a list of timedelta objects
# dataset.poses:      List of ground truth poses T_w_cam0
# dataset.camN:       Generator to load individual images from camera N
# dataset.gray:       Generator to load monochrome stereo pairs (cam0, cam1)
# dataset.rgb:        Generator to load RGB stereo pairs (cam2, cam3)
# dataset.velo:       Generator to load velodyne scans as [x,y,z,reflectance]

# Grab some data

H = None

target_dir = "/home/kluger/tmp/kitti_horizons_3/%s/" % sequence

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

M = np.matrix(np.eye(4))

for idx, gray in enumerate(iter(dataset.rgb)):
    break
    print(idx)
    if idx < 0: continue

    pose = dataset.poses[idx]
    calib0 = dataset.calib.P_rect_20
    K = np.matrix(dataset.calib.P_rect_20[0:3, 0:3])
    P = np.matrix(calib0)
    M = np.matrix(pose).I

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.imshow(gray[0])
    ax.set_title('Left Gray Image (cam0)')

    p1 = None
    p2 = None

    eh = EventHandler(fig, ax, K=K, im_centre=(gray[0].width / 2., gray[0].height / 2.))

    fig.canvas.mpl_connect('key_press_event', eh.onpress)
    fig.canvas.mpl_connect('button_press_event', eh.onclick)

    plt.show()

    # h = eh.horizon

    # H = K.T * np.matrix(h).T

    if eh.n is not None:

        H = M[0:3,0:3].I * eh.n

        H /= np.linalg.norm(H)
        # H.resize((4,1))
        print("H: ", H)

        break


H = np.matrix([[0.], [1.], [0.]])
print("H: ", H)

for idx, gray in enumerate(iter(dataset.rgb)):

    pose = dataset.poses[idx]
    calib0 = dataset.calib.P_rect_20
    K = np.matrix(dataset.calib.P_rect_20[0:3, 0:3])
    P = np.matrix(calib0)
    M = np.matrix(pose).I

    # print("P: \n", P)
    # print("M: \n", M)

    print(idx)


    # if idx == 0:

        # fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        # ax.imshow(gray[0])
        # ax.set_title('Left Gray Image (cam0)')
        #
        # p1 = None
        # p2 = None
        #
        # eh = EventHandler(fig, ax, K=K, im_centre=(gray[0].width/2., gray[0].height/2.))
        #
        # fig.canvas.mpl_connect('key_press_event', eh.onpress)
        # fig.canvas.mpl_connect('button_press_event', eh.onclick)
        #
        # plt.show()
        #
        # # h = eh.horizon
        #
        # # H = K.T * np.matrix(h).T
        #
        # H = eh.n
        #
        # H /= np.linalg.norm(H)
        # # H.resize((4,1))
        # print("H: ", H)



    # h = np.array(K.I.T*np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])*M[0:3,0:3]*H).squeeze()
    h = np.array(K.I.T*M[0:3,0:3]*H).squeeze()

    # print(h)

    hp1 = np.cross(h, np.array([1, 0, 0]))
    hp2 = np.cross(h, np.array([1, 0, -gray[0].width]))

    hp1 /= hp1[2]
    hp2 /= hp2[2]

    # print(hp1)
    # print(hp2)


    # Display some of the data
    # np.set_printoptions(precision=4, suppress=True)
    # print('\nSequence: ' + str(dataset.sequence))
    # print('\nFrame range: ' + str(dataset.frames))
    #
    # print('\nSecond ground truth pose:\n' + str(pose))

    f, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.imshow(gray[0])
    # ax.set_title('Left Gray Image (cam0)')
    plt.autoscale(False)
    ax.plot([hp1[0], hp2[0]], [hp1[1], hp2[1]], 'g-', lw=2)

    plt.axis('off')

    plt.savefig("%s/%06d.png" % (target_dir, idx), bbox_inches='tight')
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

    # plt.show()
    plt.close()

    pickle_file = "%s/%06d.pkl" % (target_dir, idx)

    with open(pickle_file, 'wb') as f:
        pickle.dump((h, H, hp1, hp2), f, -1)