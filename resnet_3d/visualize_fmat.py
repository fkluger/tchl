from resnet.resnet_plus_lstm import resnet18rnn
from datasets.kitti import KittiRawDatasetPP
from resnet.train import Config
import torch
import datetime
import os
import numpy as np
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import platform
import pykitti

from skimage import data
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
import cv2

hostname = platform.node()

# checkpoint_path = "/data/kluger/checkpoints/horizon_sequences/res18_fine/1/b32_180910-133634/009_0.001100.ckpt"
# checkpoint_path = "/data/kluger/checkpoints/horizon_sequences/res18_fine/16/b2_180911-104511/007_0.001957.ckpt"
# checkpoint_path = "/data/kluger/checkpoints/horizon_sequences/res18_fine/1/b2_180913-104954/051_0.000588.ckpt"
# checkpoint_path = "/tnt/data/kluger/checkpoints/horizon_sequences/res18_fine/1/b2_180914-104717/006_0.000127.ckpt"
# checkpoint_path = "/data/kluger/checkpoints/horizon_sequences/res18_fine/1/b2_180914-143448/003_0.001422.ckpt"
checkpoint_path = "/data/kluger/checkpoints/horizon_sequences/res18_fine/1/b2_180915-184354/017_0.001131.ckpt"
# checkpoint_path = "/data/kluger/checkpoints/horizon_sequences/res18_fine/2/b2_180916-101747/model_best.ckpt"

device = torch.device('cpu', 0)
model = resnet18rnn()

seq_length = 50
net_seq_length = 2
whole_sequence = False

checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['state_dict'], strict=False)

if 'daidalos' in hostname:
    target_base = "/tnt/data/kluger/checkpoints/horizon_sequences"
    root_dir = "/tnt/data/kluger/datasets/kitti/horizons"
    basedir = '/tnt/data/scene_understanding/KITTI/rawdata'
    csv_base = "/tnt/home/kluger/tmp/kitti_split_2"
    pdf_file = "/tnt/home/kluger/tmp/kitti_split/data_pdfs.pkl"
elif 'athene' in hostname:
    target_base = "/data/kluger/checkpoints/horizon_sequences"
    root_dir = "/phys/intern/kluger/tmp/kitti/horizons"
    csv_base = "/home/kluger/tmp/kitti_split_2"
    pdf_file = "/home/kluger/tmp/kitti_split/data_pdfs.pkl"
else:
    target_base = "/data/kluger/checkpoints/horizon_sequences"
    root_dir = "/data/kluger/datasets/kitti/horizons"
    basedir = '/data/scene_understanding/KITTI/rawdata'
    csv_base = "/home/kluger/tmp/kitti_split_2"
    pdf_file = "/home/kluger/tmp/kitti_split/data_pdfs.pkl"


train_dataset = KittiRawDatasetPP(root_dir=root_dir, pdf_file=pdf_file, augmentation=False,
                                csv_file=csv_base + "/train.csv", seq_length=seq_length)
val_dataset = KittiRawDatasetPP(root_dir=root_dir, pdf_file=pdf_file, augmentation=False,
                              csv_file=csv_base + "/val.csv", seq_length=seq_length, return_info=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=1,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                          batch_size=1,
                                          shuffle=False)


FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=10, metadata=metadata)

video_folder = "/home/kluger/tmp/kitti_videos_fmat_180918b/"
if not os.path.exists(video_folder):
    os.makedirs(video_folder)

model.eval()
with torch.no_grad():
    losses = []
    offset_losses = []
    angle_losses = []
    for idx, sample in enumerate(val_loader):

        if idx < 16: continue

        images = sample['images']
        offsets = sample['offsets']
        angles = sample['angles']

        date = sample['date'][0]
        drive = sample['drive'][0]
        start = sample['start'][0]

        dataset = pykitti.raw(basedir, date, drive)
        K = np.matrix(dataset.calib.P_rect_20[0:3, 0:3])

        print("idx ", idx)

        fig = plt.figure(figsize=(6.4, 3.0))
        # print("figsize: ", plt.rcParams['figure.figsize'])

        l1, = plt.plot([], [], 'g-', lw=4)
        l2, = plt.plot([], [], 'm-', lw=4)
        l3, = plt.plot([], [], 'y--', lw=3)

        last_image = None
        last_hor = None

        with writer.saving(fig, video_folder + "%05d.mp4" % idx, 300):
            if whole_sequence:
                output_offsets, output_angles = model(images)
            for si in range(images.shape[1]):
                fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
                    # fig.set_dpi(plt.rcParams["figure.dpi"])

                # R_imu = np.matrix(dataset.oxts[start+si].T_w_imu[0:3, 0:3])

                if not whole_sequence:
                    output_offsets, output_angles = model(images[:,si,:,:,:].unsqueeze(1))
                    offset_estm = output_offsets[0,0].numpy().squeeze()
                    angle_estm = output_angles[0,0].numpy().squeeze()
                else:
                    offset_estm = output_offsets[0,si].numpy().squeeze()
                    angle_estm = output_angles[0,si].numpy().squeeze()

                image = images.numpy()[0,si,:,:,:].transpose((1,2,0))
                width = image.shape[1]
                height = image.shape[0]

                offset = offsets[0, si].numpy().squeeze()
                angle = angles[0, si].numpy().squeeze()

                offset += 0.5
                offset *= height

                offset_estm += 0.5
                offset_estm *= height

                true_mp = np.array([width/2., offset])
                true_nv = np.array([np.sin(angle), np.cos(angle)])
                true_hl = np.array([true_nv[0], true_nv[1], -np.dot(true_nv, true_mp)])
                true_h1 = np.cross(true_hl, np.array([1, 0, 0]))
                true_h2 = np.cross(true_hl, np.array([1, 0, -width]))
                true_h1 /= true_h1[2]
                true_h2 /= true_h2[2]

                estm_mp = np.array([width/2., offset_estm])
                estm_nv = np.array([np.sin(angle_estm), np.cos(angle_estm)])
                estm_hl = np.array([estm_nv[0], estm_nv[1], -np.dot(estm_nv, estm_mp)])
                estm_h1 = np.cross(estm_hl, np.array([1, 0, 0]))
                estm_h2 = np.cross(estm_hl, np.array([1, 0, -width]))
                estm_h1 /= estm_h1[2]
                estm_h2 /= estm_h2[2]


                # print("true: %.2f px, %.2f deg" % (offset, angle*180./np.pi))
                # print("estm: %.2f px, %.2f deg" % (offset_estm, angle_estm*180./np.pi))

                # plt.figure()

                # fig.set_size_inches(image.shape[1], image.shape[0], forward=True)
                plt.imshow(image)
                plt.axis('off')
                plt.autoscale(False)
                # plt.tight_layout()
                # plt.plot([true_h1[0], true_h2[0]], [true_h1[1], true_h2[1]], 'g-', lw=4)
                # plt.plot([estm_h1[0], estm_h2[0]], [estm_h1[1], estm_h2[1]], 'm-', lw=4)

                l1.set_data([true_h1[0], true_h2[0]], [true_h1[1], true_h2[1]])
                l2.set_data([estm_h1[0], estm_h2[0]], [estm_h1[1], estm_h2[1]])

                plt.suptitle("true: %.1f px, %.1f deg --- error: %.1f px, %.1f deg" %
                             (offset, angle*180./np.pi, np.abs(offset-offset_estm),
                              np.abs(angle-angle_estm)*180./np.pi), family='monospace', y=0.9)

                if last_hor is not None and last_image is not None:

                    img_left, img_right = map(rgb2gray, (last_image, image))

                    descriptor_extractor = ORB()

                    descriptor_extractor.detect_and_extract(img_left)
                    keypoints_left = descriptor_extractor.keypoints
                    descriptors_left = descriptor_extractor.descriptors

                    descriptor_extractor.detect_and_extract(img_right)
                    keypoints_right = descriptor_extractor.keypoints
                    descriptors_right = descriptor_extractor.descriptors

                    matches = match_descriptors(descriptors_left, descriptors_right,
                                                cross_check=True)

                    # Estimate the epipolar geometry between the left and right image.

                    F_model, inliers = ransac((keypoints_left[matches[:, 0]],
                                             keypoints_right[matches[:, 1]]),
                                            FundamentalMatrixTransform, min_samples=8,
                                            residual_threshold=1, max_trials=5000)

                    inlier_keypoints_left = keypoints_left[matches[inliers, 0]]
                    inlier_keypoints_right = keypoints_right[matches[inliers, 1]]

                    # Visualize the results.

                    # fig, ax = plt.subplots(nrows=1, ncols=1)
                    #
                    # plt.gray()
                    #
                    # plot_matches(ax, img_left, img_right, keypoints_left, keypoints_right,
                    #              matches[inliers], only_matches=True)
                    #
                    # plt.show()

                    F = np.matrix(F_model.params)
                    E = K.T * F * K

                    R1, R2, t = cv2.decomposeEssentialMat(np.array(E))

                    print("last_hor: ", last_hor)
                    n = K.T * np.matrix(last_hor).T
                    n /= np.linalg.norm(n)
                    nr1 = R1 * n
                    nr2 = R2 * n

                    e1 = np.abs(np.array(n.T * nr1).squeeze())
                    e2 = np.abs(np.array(n.T * nr2).squeeze())

                    if e1 > e2:
                        nr = nr1
                    else:
                        nr = nr2

                    hr = np.array(K.T.I * nr).squeeze()
                    hr /= np.linalg.norm(hr[0:2])
                    print("est. hor: ", hr)
                    print("true hor: ", true_hl)

                    print("Number of matches:", matches.shape[0])
                    print("Number of inliers:", inliers.sum())

                    last_hor = hr

                    rot_h1 = np.cross(hr, np.array([1, 0, 0]))
                    rot_h2 = np.cross(hr, np.array([1, 0, -width]))
                    rot_h1 /= rot_h1[2]
                    rot_h2 /= rot_h2[2]

                    l3.set_data([rot_h1[0], rot_h2[0]], [rot_h1[1], rot_h2[1]])

                else:
                    last_hor = true_hl.copy()

                last_image = image.copy()
                # last_R_imu = R_imu

                # plt.show()
                writer.grab_frame()
                # plt.clf()

        plt.close()
