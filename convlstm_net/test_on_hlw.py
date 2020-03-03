import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from convlstm_net.resnet_plus_lstm import resnet18rnn
import torch
from torchvision import transforms
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import glob
import math
import sklearn.metrics
from utilities.auc import *
from hlw.hlw import HLWDataset, WIDTH,  HEIGHT

np.seterr(all='raise')

def calc_horizon_leftright(width, height):
    wh = 0.5 * width

    def f(offset, angle):
        term2 = wh * torch.tan(torch.clamp(angle, -math.pi/3., math.pi/3.)).cpu().detach().numpy().squeeze()
        offset = offset.cpu().detach().numpy().squeeze()
        angle = angle.cpu().detach().numpy().squeeze()
        return height * offset + height * 0.5 + term2, height * offset + height * 0.5 - term2

    return f

parser = argparse.ArgumentParser(description='')
parser.add_argument('--load', '-l', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--set', '-s', default='val', type=str, metavar='PATH', help='')
parser.add_argument('--batch', default=1, type=int, metavar='N', help='')
parser.add_argument('--skip', dest='skip', action='store_true', help='')
parser.add_argument('--fc', dest='fc', action='store_true', help='')
parser.add_argument('--lstm_state_reduction', default=4., type=float, metavar='S', help='random subsampling factor')
parser.add_argument('--lstm_depth', default=1, type=int, metavar='S', help='random subsampling factor')
parser.add_argument('--cpu', dest='cpu', action='store_true', help='')
parser.add_argument('--gpu', default='0', type=str, metavar='DS', help='dataset')
parser.add_argument('--convlstm', dest='convlstm', action='store_true', help='')
parser.add_argument('--simple_skip', dest='simple_skip', action='store_true', help='')

args = parser.parse_args()

checkpoint_path = args.load if not (args.load == '') else None

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if args.cpu:
    device = torch.device('cpu', 0)
else:
    device = torch.device('cuda', 0)

downscale = 2.

if checkpoint_path is not None:

    model = resnet18rnn(use_fc=args.fc, use_convlstm=args.convlstm, lstm_skip=args.skip,
                        lstm_mem=1, lstm_depth=args.lstm_depth, lstm_state_reduction=args.lstm_state_reduction,
                        lstm_simple_skip=args.simple_skip).to(device)

    load_from_path = checkpoint_path
    print("load weights from ", load_from_path)
    checkpoint = torch.load(load_from_path, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()

im_width = int(WIDTH/downscale)
im_height = int(HEIGHT/downscale)

calc_hlr = calc_horizon_leftright(im_width, im_height)

all_errors = []
all_angular_errors = []
max_errors = []
image_count = 0

percs = [50, 80, 90, 95, 99]
print("percentiles: ", percs)

error_grads = []

root_dir = "/data/scene_understanding/HLW/"

pixel_mean = [0.469719773, 0.462005855, 0.454649294]

tfs = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=pixel_mean, std=[1., 1., 1.]),
        ])


dataset = HLWDataset(root_dir, set=args.set, augmentation=False, transform=tfs, scale=1./downscale)

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

                width = im_width
                height = im_height

                offset = offsets[bi, si].detach().numpy().squeeze()
                angle = angles[bi, si].detach().numpy().squeeze()

                yl, yr = calc_hlr(offsets[bi, si], angles[bi, si])

                if checkpoint_path is not None:

                    offset_estm = output_offsets[bi,si].detach().cpu().numpy().squeeze()
                    angle_estm = output_angles[bi,si].detach().cpu().numpy().squeeze()

                    yle, yre = calc_hlr(output_offsets[bi, si], output_angles[bi, si])

                    err1 = np.abs((yl - yle) / height)
                    err2 = np.abs((yr - yre) / height)

                    err = np.maximum(err1, err2)

                    print("error: %.3f" % err)

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
print("AUC: ", auc)
print("MSE: ", np.mean(error_arr))

plt.figure()
plt.plot(plot_points[:,0], plot_points[:,1], 'b-')
plt.xlim(0, 0.25)
plt.ylim(0, 1.0)
plt.show()





