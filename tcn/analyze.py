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
from torchvision import transforms
os.environ["CUDA_VISIBLE_DEVICES"]="0"

hostname = platform.node()

# checkpoint_path = "/data/kluger/checkpoints/horizon_sequences/kitti/res18_fine/d1/8/b4_181119-175638"
# checkpoint_path = "/data/kluger/checkpoints/horizon_sequences/kitti/res18_fine/d1/1/b32_181106-220154"
checkpoint_path = "/data/kluger/checkpoints/horizon_sequences/kitti/res18_fine/d1/4/b8_181120-185938"

result_folder = os.path.join(checkpoint_path, "results")
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

device = torch.device('cuda', 0)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', 0)

model = resnet18rnn(load=False, use_fc=True, use_convlstm=True).to(device)

seq_length = 4
batch_size = 1
whole_sequence = True

checkpoint = torch.load(os.path.join(checkpoint_path, "model_best.ckpt"), map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['state_dict'], strict=False)

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

pixel_mean = [0.362365, 0.377767, 0.366744]

tfs = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=pixel_mean, std=[1., 1., 1.]),
        ])

test_dataset = KittiRawDatasetPP(root_dir=root_dir, pdf_file=pdf_file, augmentation=False, transform=tfs,
                                csv_file=csv_base + "/test.csv", seq_length=seq_length, fill_up=False)
val_dataset = KittiRawDatasetPP(root_dir=root_dir, pdf_file=pdf_file, augmentation=False, transform=tfs,
                              csv_file=csv_base + "/val.csv", seq_length=seq_length, fill_up=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

all_errors = []

model.eval()
with torch.no_grad():
    losses = []
    offset_losses = []
    angle_losses = []
    for idx, sample in enumerate(val_loader):

        images = sample['images'].to(device)
        offsets = sample['offsets']
        angles = sample['angles']

        print("idx ", idx)
        all_offsets = []
        all_offsets_estm = []

        if whole_sequence:
            output_offsets, output_angles = model(images)

        for si in range(images.shape[1]):

            if not whole_sequence:
                output_offsets, output_angles = model(images[:, si, :, :, :].unsqueeze(1))
                offset_estm_batch = output_offsets[:, 0].cpu().numpy().squeeze()
                angle_estm_batch = output_angles[:, 0].cpu().numpy().squeeze()
            else:
                offset_estm_batch = output_offsets[:, si].cpu().numpy().squeeze()
                angle_estm_batch = output_angles[:, si].cpu().numpy().squeeze()

            for bi in range(offset_estm_batch.shape[0] if batch_size > 1 else 1):

                offset_estm = offset_estm_batch[bi] if batch_size > 1 else offset_estm_batch
                angle_estm = angle_estm_batch[bi] if batch_size > 1 else angle_estm_batch


                width = images.shape[4]
                height = images.shape[3]

                offset = offsets[bi, si].numpy().squeeze()
                angle = angles[bi, si].numpy().squeeze()

                all_offsets += [offset.copy()]
                all_offsets_estm += [offset_estm.copy()]

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


        # plt.figure()
        # x = np.arange(0, len(all_offsets))
        # all_offsets = np.array(all_offsets)
        # all_offsets_estm = np.array(all_offsets_estm)
        # print(all_offsets)
        # plt.plot(x, all_offsets, 'b-')
        # plt.plot(x, all_offsets_estm, 'r-')
        # plt.ylim(-.4, .4)
        # plt.savefig(os.path.join(result_folder, "offsets_%03d.png" % idx), dpi=300)
        # plt.savefig(os.path.join(result_folder, "offsets_%03d.svg" % idx), dpi=300)
        # plt.close()

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
plt.savefig("/home/kluger/tmp/kitti_auc_7.png", dpi=300)
plt.savefig("/home/kluger/tmp/kitti_auc_7.svg", dpi=300)