from resnet.resnet_plus_lstm import resnet18rnn
from datasets.hlw import HLWDataset, WIDTH,  HEIGHT
from resnet.train import Config
import torch
import datetime
from torchvision import transforms
import os
import numpy as np
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import platform
hostname = platform.node()
import argparse
import sklearn.metrics

parser = argparse.ArgumentParser(description='')
parser.add_argument('--load', '-l', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--set', '-s', default='val', type=str, metavar='PATH', help='')
parser.add_argument('--dev', '-d', default=0, type=int, metavar='PATH', help='')
parser.add_argument('--batch', '-b', default=16, type=int, metavar='PATH', help='')

args = parser.parse_args()

print(args)

checkpoint_path = args.load if not (args.load == '') else None
set_type = args.set


os.environ["CUDA_VISIBLE_DEVICES"]="%d" % args.dev
device = torch.device('cuda' if args.dev >= 0 else 'cpu', 0)

model = resnet18rnn(regional_pool=(3,3))
# model = resnet18rnn()
checkpoint = torch.load(os.path.join(checkpoint_path, "model_best.ckpt"), map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['state_dict'], strict=False)
model.to(device)
model.eval()

if 'daidalos' in hostname:
    target_base = "/tnt/data/kluger/checkpoints/horizon_sequences"
    root_dir = "/tnt/data/scene_understanding/HLW/"
elif 'athene' in hostname:
    target_base = "/data/kluger/checkpoints/horizon_sequences"
    root_dir = "/data/scene_understanding/HLW/"
else:
    target_base = "/data/kluger/checkpoints/horizon_sequences"
    root_dir = "/data/scene_understanding/HLW/"

pixel_mean = [0.469719773, 0.462005855, 0.454649294]

tfs = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=pixel_mean, std=[1., 1., 1.]),
        ])


dataset = HLWDataset(root_dir, set=args.set, augmentation=False, transform=tfs)

loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=args.batch,
                                           shuffle=False)

all_errors = []

with torch.no_grad():
    losses = []
    offset_losses = []
    angle_losses = []
    for idx, sample in enumerate(loader):

        images = sample['images'].to(device)
        offsets = sample['offsets']
        angles = sample['angles']

        print("batch %d / %d " % (idx+1, len(loader)))
        all_offsets = []
        all_offsets_estm = []

        output_offsets, output_angles = model(images)

        for bi in range(images.shape[0]):
            for si in range(images.shape[1]):

                # image = images.numpy()[0,si,:,:,:].transpose((1,2,0))
                width = WIDTH #image.shape[1]
                height = HEIGHT #image.shape[0]

                offset = offsets[bi, si].detach().numpy().squeeze()
                angle = angles[bi, si].detach().numpy().squeeze()

                all_offsets += [-offset.copy()]

                offset += 0.5
                offset *= height

                true_mp = np.array([width/2., offset])
                true_nv = np.array([np.sin(angle), np.cos(angle)])
                true_hl = np.array([true_nv[0], true_nv[1], -np.dot(true_nv, true_mp)])
                true_h1 = np.cross(true_hl, np.array([1, 0, 0]))
                true_h2 = np.cross(true_hl, np.array([1, 0, -width]))
                true_h1 /= true_h1[2]
                true_h2 /= true_h2[2]

                if checkpoint_path is not None:

                    offset_estm = output_offsets[bi,si].detach().cpu().numpy().squeeze()
                    angle_estm = output_angles[bi,si].detach().cpu().numpy().squeeze()

                    all_offsets_estm += [-offset_estm.copy()]

                    offset_estm += 0.5
                    offset_estm *= height

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

                    print("%.3f" % err)

                    all_errors.append(err)


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
# print("errors: \n", error_arr)
# print(error_arr_idx)
print("mean error: ", np.mean(error_arr))

plt.figure()
plt.plot(plot_points[:,0], plot_points[:,1], 'b-')
plt.xlim(0, 0.25)
plt.ylim(0, 1.0)
plt.savefig("/home/kluger/tmp/hlw_auc.png", dpi=300)
plt.savefig("/home/kluger/tmp/hlw_auc.svg", dpi=300)





