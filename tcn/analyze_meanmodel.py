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
import sklearn.metrics
import os

hostname = platform.node()

seq_length = 1
batch_size = 64
whole_sequence = True

if 'daidalos' in hostname:
    target_base = "/tnt/data/kluger/checkpoints/horizon_sequences"
    root_dir = "/tnt/data/kluger/datasets/kitti/horizons"
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
    csv_base = "/home/kluger/tmp/kitti_split_2"
    pdf_file = "/home/kluger/tmp/kitti_split/data_pdfs.pkl"


train_dataset = KittiRawDatasetPP(root_dir=root_dir, pdf_file=pdf_file, augmentation=False,
                                csv_file=csv_base + "/train.csv", seq_length=seq_length, fill_up=False)
val_dataset = KittiRawDatasetPP(root_dir=root_dir, pdf_file=pdf_file, augmentation=False,
                              csv_file=csv_base + "/val.csv", seq_length=seq_length, fill_up=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

all_errors = []

with torch.no_grad():
    losses = []
    offset_losses = []
    angle_losses = []
    for idx, sample in enumerate(val_loader):

        images = sample['images']
        offsets = sample['offsets']
        angles = sample['angles']

        print("idx ", idx)

        for si in range(images.shape[1]):

            for bi in range(offsets.shape[0]):

                offset_estm = -0.039186
                angle_estm = 0.011849


                width = images.shape[4]
                height = images.shape[3]

                offset = offsets[bi, si].numpy().squeeze()
                angle = angles[bi, si].numpy().squeeze()

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

                err1 = np.minimum(np.linalg.norm(estm_h1 - true_h1), np.linalg.norm(estm_h1 - true_h2))
                err2 = np.minimum(np.linalg.norm(estm_h2 - true_h1), np.linalg.norm(estm_h2 - true_h2))

                err = np.maximum(err1,err2) / height

                print("%.3f" % err, end=", ")

                all_errors.append(err)
        print(" ")

error_arr = np.array(all_errors)
error_arr_idx = np.argsort(error_arr)
error_arr = np.sort(error_arr)
num_values = len(all_errors)

plot_points = np.zeros((num_values,2))

err_cutoff = 0.25

for i in range(num_values):
    fraction = (i+1) * 1.0/num_values
    value = error_arr[i]
    plot_points[i,1] = fraction
    plot_points[i,0] = value
    if i > 0:
        lastvalue = error_arr[i-1]
        if lastvalue < err_cutoff and value > err_cutoff:
            midfraction = (lastvalue*plot_points[i-1,1] + value*fraction) / (value+lastvalue)

if plot_points[-1,0] < err_cutoff:
    plot_points = np.vstack([plot_points, np.array([err_cutoff,1])])
else:
    plot_points = np.vstack([plot_points, np.array([err_cutoff,midfraction])])

sorting = np.argsort(plot_points[:,0])
plot_points = plot_points[sorting,:]

auc = sklearn.metrics.auc(plot_points[plot_points[:,0]<=err_cutoff,0], plot_points[plot_points[:,0]<=err_cutoff,1])
auc =  auc / err_cutoff
print("auc: ", auc)
print("errors: \n", error_arr)
print(error_arr_idx)
print("mean error: ", np.mean(error_arr))

plt.figure()
plt.plot(plot_points[:,0], plot_points[:,1], 'b-')
plt.xlim(0, 0.25)
plt.ylim(0, 1.0)
plt.savefig("/home/kluger/tmp/kitti_meanmodel_auc.png", dpi=300)
plt.savefig("/home/kluger/tmp/kitti_meanmodel_auc.svg", dpi=300)