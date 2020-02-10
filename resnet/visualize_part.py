import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from resnet.resnet_plus_lstm import resnet18rnn
from datasets.kitti import KittiRawDatasetPP, WIDTH,  HEIGHT
from resnet.train import Config
from utilities.tee import Tee
import torch
from torch import nn
import datetime
from torchvision import transforms
import os
import numpy as np
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import platform
import logging
logger = logging.getLogger('matplotlib.animation')
logger.setLevel(logging.DEBUG)
hostname = platform.node()
import argparse
import glob
from mpl_toolkits.mplot3d import Axes3D
import contextlib
import math
import sklearn.metrics
from utilities.auc import *

np.seterr(all='raise')

def calc_horizon_leftright(width, height):
    wh = 0.5 * width#*1./height

    def f(offset, angle):
        term2 = wh * torch.tan(torch.clamp(angle, -math.pi/3., math.pi/3.)).cpu().detach().numpy().squeeze()
        offset = offset.cpu().detach().numpy().squeeze()
        angle = angle.cpu().detach().numpy().squeeze()
        return height * offset + height * 0.5 + term2, height * offset + height * 0.5 - term2

    return f

parser = argparse.ArgumentParser(description='')
parser.add_argument('--load1', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--load2', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--video', dest='video', action='store_true', help='')
parser.add_argument('--seqlength', default=10000, type=int, metavar='N', help='')
parser.add_argument('--cpid', default=-1, type=int, metavar='N', help='')
parser.add_argument('--split', default=5, type=int, metavar='N', help='')
parser.add_argument('--lstm_mem', default=0, type=int, metavar='N', help='')
parser.add_argument('--skip', dest='skip', action='store_true', help='')
parser.add_argument('--fc', dest='fc', action='store_true', help='')
parser.add_argument('--lstm_state_reduction', default=1., type=float, metavar='S', help='random subsampling factor')
parser.add_argument('--lstm_depth', default=1, type=int, metavar='S', help='random subsampling factor')
parser.add_argument('--trainable_lstm_init', dest='trainable_lstm_init', action='store_true', help='')
parser.add_argument('--cpu', dest='cpu', action='store_true', help='')
parser.add_argument('--gpu', default='0', type=str, metavar='DS', help='dataset')
parser.add_argument('--bias', dest='bias', action='store_true', help='')
parser.add_argument('--convlstm', dest='convlstm', action='store_true', help='')
parser.add_argument('--tee', dest='tee', action='store_true', help='')
parser.add_argument('--simple_skip', dest='simple_skip', action='store_true', help='')
parser.add_argument('--layernorm', dest='layernorm', action='store_true', help='')
parser.add_argument('--lstm_leakyrelu', dest='lstm_leakyrelu', action='store_true', help='')
parser.add_argument('--set', '-s', default=None, type=str, metavar='PATH', help='')

parser.add_argument('--date', default='0', type=str, metavar='DS', help='dataset')
parser.add_argument('--drive', default='0', type=str, metavar='DS', help='dataset')
parser.add_argument('--start', default=0, type=int, metavar='DS', help='dataset')
parser.add_argument('--end', default=0, type=int, metavar='DS', help='dataset')

args = parser.parse_args()
set_type = args.set

checkpoint_path_1 = args.load1 if not (args.load1 == '') else None
checkpoint_path_2 = args.load2 if not (args.load2 == '') else None


if 'daidalos' in hostname:
    target_base = "/tnt/data/kluger/checkpoints/horizon_sequences"
    root_dir = "/tnt/data/kluger/datasets/kitti/horizons"
    csv_base = "/tnt/home/kluger/tmp/kitti_split_%d" % args.split
    pdf_file = "/tnt/home/kluger/tmp/kitti_split/data_pdfs.pkl"
elif 'athene' in hostname:
    target_base = "/data/kluger/checkpoints/horizon_sequences"
    root_dir = "/phys/intern/kluger/tmp/kitti/horizons"
    csv_base = "/home/kluger/tmp/kitti_split_%d" % args.split
    pdf_file = "/home/kluger/tmp/kitti_split/data_pdfs.pkl"
elif 'hekate' in hostname:
    target_base = "/data/kluger/checkpoints/horizon_sequences"
    root_dir = "/phys/ssd/kitti/horizons"
    csv_base = "/home/kluger/tmp/kitti_split_%d" % args.split
    pdf_file = "/home/kluger/tmp/kitti_split/data_pdfs.pkl"
else:
    target_base = "/data/kluger/checkpoints/horizon_sequences"
    root_dir = "/data/kluger/datasets/kitti/horizons"
    csv_base = "/home/kluger/tmp/kitti_split_%d" % args.split
    pdf_file = "/home/kluger/tmp/kitti_split/data_pdfs.pkl"


downscale = 2.
if downscale > 1:
    root_dir += "_s%.3f" % (1./downscale)

root_dir += "_ema%.3f" % 0.1

if checkpoint_path_1 is None and checkpoint_path_2 is None:
    result_folder = "/home/kluger/tmp/kitti_horizon_videos_999/"
elif checkpoint_path_2 is None:
    result_folder = os.path.join(checkpoint_path_1, "results/")
else:
    cp2_tail = os.path.split(checkpoint_path_2)[1]
    result_folder = os.path.join(checkpoint_path_1, "vs_%s/results/" % cp2_tail)

if set_type is not None:
    result_folder = os.path.join(result_folder, set_type)
    csv_file = csv_base + "/%s.csv" % set_type
else:
    csv_file = None

print(result_folder)

if not os.path.exists(result_folder):
    os.makedirs(result_folder)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if args.cpu:
    device = torch.device('cpu', 0)
else:
    device = torch.device('cuda', 0)
# device = torch.device('cpu', 0)

seq_length = args.seqlength


baseline_angle = 0.013490
baseline_offset = -0.036219

ymin = -0.2
ymax = 0.2

if checkpoint_path_1 is not None:

    if args.cpid == -1:
        cp_path = os.path.join(checkpoint_path_1, "model_best.ckpt")
    else:
        cp_path_reg = os.path.join(checkpoint_path_1, "%03d_*.ckpt" % args.cpid)
        cp_path = glob.glob(cp_path_reg)[0]

    load_from_path = cp_path
    print("load weights from ", load_from_path)
    checkpoint = torch.load(load_from_path, map_location=lambda storage, loc: storage)

    model_args = checkpoint['args']

    model1 = resnet18rnn(regional_pool=None, use_fc=model_args.fc_layer, use_convlstm=model_args.conv_lstm,
                         lstm_bias=model_args.bias, width=WIDTH, height=HEIGHT,
                         trainable_lstm_init=model_args.trainable_lstm_init,
                         conv_lstm_skip=False, confidence=False, second_head=False,
                         relu_lstm=False, second_head_fc=False, lstm_bn=False, lstm_skip=model_args.skip,
                         lstm_peephole=False, lstm_mem=args.lstm_mem, lstm_depth=model_args.lstm_depth,
                         lstm_state_reduction=model_args.lstm_state_reduction, lstm_simple_skip=args.simple_skip,
                         layernorm=False, lstm_leakyrelu=False).to(device)

    model1.load_state_dict(checkpoint['state_dict'], strict=True)
    model1.eval()

if checkpoint_path_2 is not None:

    if args.cpid == -1:
        cp_path = os.path.join(checkpoint_path_2, "model_best.ckpt")
    else:
        cp_path_reg = os.path.join(checkpoint_path_2, "%03d_*.ckpt" % args.cpid)
        cp_path = glob.glob(cp_path_reg)[0]

    load_from_path = cp_path
    print("load weights from ", load_from_path)
    checkpoint = torch.load(load_from_path, map_location=lambda storage, loc: storage)

    model_args = checkpoint['args']

    model2 = resnet18rnn(regional_pool=None, use_fc=model_args.fc_layer, use_convlstm=model_args.conv_lstm,
                         lstm_bias=model_args.bias, width=WIDTH, height=HEIGHT,
                         trainable_lstm_init=model_args.trainable_lstm_init,
                         conv_lstm_skip=False, confidence=False, second_head=False,
                         relu_lstm=False, second_head_fc=False, lstm_bn=False, lstm_skip=model_args.skip,
                         lstm_peephole=False, lstm_mem=args.lstm_mem, lstm_depth=model_args.lstm_depth,
                         lstm_state_reduction=model_args.lstm_state_reduction, lstm_simple_skip=args.simple_skip,
                         layernorm=False, lstm_leakyrelu=False).to(device)

    model2.load_state_dict(checkpoint['state_dict'], strict=True)
    model2.eval()


pixel_mean = [0.362365, 0.377767, 0.366744]

tfs_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=pixel_mean, std=[1., 1., 1.]),
        ])

im_width = int(WIDTH/downscale)
im_height = int(HEIGHT/downscale)

calc_hlr = calc_horizon_leftright(im_width, im_height)

dataset = KittiRawDatasetPP(root_dir=root_dir, pdf_file=pdf_file, augmentation=False, im_height=im_height,
                            im_width=im_width, return_info=True, get_split_data=False,
                            csv_file=csv_file, seq_length=seq_length, fill_up=False, transform=tfs_val,
                            single_sequence=(args.date, args.drive, args.start, args.end))

loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=1,
                                     shuffle=False)

result_folder = os.path.join(result_folder, "%s_%s_%d_%d/" % (args.date, args.drive, args.start, args.end))
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

if args.tee:
    log_file = os.path.join(result_folder, "log")
    log = Tee(os.path.join(result_folder, log_file), "w", file_only=False)

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!', codec='vp9')
writer = FFMpegWriter(fps=10, metadata=metadata, codec='libx264', extra_args=['-intra'])


video_folder = result_folder
if not os.path.exists(video_folder):
    os.makedirs(video_folder)

png_folder = os.path.join(video_folder, "png")
if not os.path.exists(png_folder):
    os.makedirs(png_folder)
svg_folder = os.path.join(video_folder, "svg")
if not os.path.exists(svg_folder):
    os.makedirs(svg_folder)

cam = None

all_errors_1 = []
all_angular_errors_1 = []
error_grads_1 = []
all_errors_2 = []
all_angular_errors_2 = []
error_grads_2 = []
image_count = 0

with torch.no_grad():
    losses_1 = []
    offset_losses_1 = []
    angle_losses_1 = []
    losses_2 = []
    offset_losses_2 = []
    angle_losses_2 = []
    for idx, sample in enumerate(loader):

        images = sample['images']
        offsets = sample['offsets']
        angles = sample['angles']
        Gs = sample['G'][0]
        padding = sample['padding']
        scale = sample['scale'].numpy()
        K = np.matrix(sample['K'][0])

        print("idx ", idx)
        all_offsets = []
        all_offsets_estm_1 = []
        all_offsets_estm_2 = []
        all_angles = []
        all_angles_estm_1 = []
        all_angles_estm_2 = []

        all_errors_per_sequence_1 = []
        all_angular_errors_per_sequence_1 = []
        all_errors_per_sequence_2 = []
        all_angular_errors_per_sequence_2 = []

        if args.video:
            # fig = plt.figure(figsize=(6.4, 3.6))

            fig, (plt1, plt2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(6.4, 3.6))

            # plt1 = plt.subplot(2, 1, 1, aspect=HEIGHT*1./WIDTH)
            # plt2 = plt.subplot(2, 1, 2)
            # plt2.set_ylim([ymin, ymax])
            plt2.set_xlim([0, images.shape[1]])
            # plt2.autoscale(False)

            lt, = plt1.plot([], [], '-', lw=4, c='#ffffff')
            l1, = plt1.plot([], [], '-', lw=4, c='#99C000')
            l2, = plt1.plot([], [], '--', lw=4, c='#FDCA00')

        with writer.saving(fig, video_folder + "%05d.mp4" % (idx), 300) if args.video else contextlib.suppress():

            if checkpoint_path_1 is not None:
                output_offsets_1, output_angles_1 = model1(images.to(device))
            if checkpoint_path_2 is not None:
                output_offsets_2, output_angles_2 = model2(images.to(device))

            for si in range(images.shape[1]):

                # print(".", end="")

                image_count += 1

                image = images.numpy()[0,si,:,:,:].transpose((1,2,0))
                width = image.shape[1]
                height = image.shape[0]

                offset = offsets[0, si].detach().numpy().squeeze()
                angle = angles[0, si].detach().numpy().squeeze()
                yl, yr = calc_hlr(offsets[0, si], angles[0, si])
                all_offsets += [-offset.copy()]
                all_angles += [angle.copy()]
                offset += 0.5
                offset *= height
                true_mp = np.array([width/2., offset])
                true_nv = np.array([np.sin(angle), np.cos(angle)])
                true_hl = np.array([true_nv[0], true_nv[1], -np.dot(true_nv, true_mp)])
                true_h1 = np.cross(true_hl, np.array([1, 0, 0]))
                true_h2 = np.cross(true_hl, np.array([1, 0, -width]))
                true_h1 /= true_h1[2]
                true_h2 /= true_h2[2]
                h1_ = true_h1/scale - np.array([padding[0], padding[2], 1/scale-1])
                h2_ = true_h2/scale - np.array([padding[0], padding[2], 1/scale-1])
                h_ = np.cross(h1_, h2_)
                Gt = K.T * np.matrix(h_).T
                Gt /= np.linalg.norm(Gt)

                if args.video:
                    # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
                    image[:, :, 0] += pixel_mean[0]
                    image[:, :, 1] += pixel_mean[1]
                    image[:, :, 2] += pixel_mean[2]
                    plt1.imshow(image)
                    plt1.axis('off')
                    plt1.autoscale(False)
                    lt.set_data([true_h1[0], true_h2[0]], [true_h1[1], true_h2[1]])

                    plt2.clear()
                    plt2.set_xlim([0, images.shape[1]])

                if checkpoint_path_1 is not None:

                    offset_estm = output_offsets_1[0, si].cpu().detach().numpy().squeeze()
                    angle_estm = output_angles_1[0, si].cpu().detach().numpy().squeeze()

                    yle, yre = calc_hlr(output_offsets_1[0, si], output_angles_1[0, si])

                    all_offsets_estm_1 += [-offset_estm.copy()]
                    all_angles_estm_1 += [angle_estm.copy()]

                    offset_estm += 0.5
                    offset_estm *= height

                    estm_mp = np.array([width/2., offset_estm])
                    estm_nv = np.array([np.sin(angle_estm), np.cos(angle_estm)])
                    estm_hl = np.array([estm_nv[0], estm_nv[1], -np.dot(estm_nv, estm_mp)])
                    estm_1_h1 = np.cross(estm_hl, np.array([1, 0, 0]))
                    estm_1_h2 = np.cross(estm_hl, np.array([1, 0, -width]))
                    estm_1_h1 /= estm_1_h1[2]
                    estm_1_h2 /= estm_1_h2[2]

                    h1_ = estm_1_h1 / scale - np.array([padding[0], padding[2], 1 / scale - 1])
                    h2_ = estm_1_h2 / scale - np.array([padding[0], padding[2], 1 / scale - 1])
                    h_ = np.cross(h1_, h2_)
                    Ge = K.T * np.matrix(h_).T
                    Ge /= np.linalg.norm(Ge)
                    G1 = np.matrix(Gs[si]).T
                    G1 /= np.linalg.norm(G1)

                    err1 = (yl-yle) / height
                    err2 = (yr-yre) / height
                    err = np.maximum(err1, err2) / height
                    if np.abs(err1) > np.abs(err2):
                        err = err1
                    else:
                        err = err2
                    all_errors_1.append(err)
                    all_errors_per_sequence_1.append(err)

                    error_arr_1 = np.abs(np.array(all_errors_1))
                    MSE1 = np.mean(np.square(error_arr_1))
                    if len(all_errors_1) > 1:
                        auc_1, _ = calc_auc(error_arr_1, cutoff=0.25)
                    else:
                        auc_1 = 0


                    try:
                        angular_error = np.abs((np.arccos(np.clip(np.abs(np.dot(Ge.T, G1)), 0, 1)) * 180 / np.pi)[0, 0])
                    except:
                        print(Ge)
                        print(G1)
                        print(np.dot(Ge.T, G1))
                        exit(0)

                    all_angular_errors_1.append(angular_error)
                    all_angular_errors_per_sequence_1.append(angular_error)

                    if args.video:
                        l1.set_data([estm_1_h1[0], estm_1_h2[0]], [estm_1_h1[1], estm_1_h2[1]])

                if checkpoint_path_2 is not None:

                    offset_estm = output_offsets_2[0, si].cpu().detach().numpy().squeeze()
                    angle_estm = output_angles_2[0, si].cpu().detach().numpy().squeeze()

                    yle, yre = calc_hlr(output_offsets_2[0, si], output_angles_2[0, si])

                    all_offsets_estm_2 += [-offset_estm.copy()]
                    all_angles_estm_2 += [angle_estm.copy()]

                    offset_estm += 0.5
                    offset_estm *= height

                    estm_mp = np.array([width/2., offset_estm])
                    estm_nv = np.array([np.sin(angle_estm), np.cos(angle_estm)])
                    estm_hl = np.array([estm_nv[0], estm_nv[1], -np.dot(estm_nv, estm_mp)])
                    estm_2_h1 = np.cross(estm_hl, np.array([1, 0, 0]))
                    estm_2_h2 = np.cross(estm_hl, np.array([1, 0, -width]))
                    estm_2_h1 /= estm_2_h1[2]
                    estm_2_h2 /= estm_2_h2[2]

                    h1_ = estm_2_h1 / scale - np.array([padding[0], padding[2], 1 / scale - 1])
                    h2_ = estm_2_h2 / scale - np.array([padding[0], padding[2], 1 / scale - 1])
                    h_ = np.cross(h1_, h2_)
                    Ge = K.T * np.matrix(h_).T
                    Ge /= np.linalg.norm(Ge)
                    G2 = np.matrix(Gs[si]).T
                    G2 /= np.linalg.norm(G2)

                    err1 = (yl-yle) / height
                    err2 = (yr-yre) / height
                    err = np.maximum(err1, err2) / height
                    if np.abs(err1) > np.abs(err2):
                        err = err1
                    else:
                        err = err2
                    all_errors_2.append(err)
                    all_errors_per_sequence_2.append(err)

                    try:
                        angular_error = np.abs((np.arccos(np.clip(np.abs(np.dot(Ge.T, G2)), 0, 1)) * 180 / np.pi)[0, 0])
                    except:
                        print(Ge)
                        print(G2)
                        print(np.dot(Ge.T, G2))
                        exit(0)

                    all_angular_errors_2.append(angular_error)
                    all_angular_errors_per_sequence_2.append(angular_error)

                    error_arr_2 = np.abs(np.array(all_errors_2))
                    MSE2 = np.mean(np.square(error_arr_2))
                    if len(all_errors_2) > 1:
                        auc_2, _ = calc_auc(error_arr_2, cutoff=0.25)
                    else:
                        auc_2 = 0

                    if args.video:
                        l2.set_data([estm_2_h1[0], estm_2_h2[0]], [estm_2_h1[1], estm_2_h2[1]])

                if args.video:
                    plt2.plot(np.arange(0, len(all_offsets)), np.array(all_offsets), '-', c='#000000', lw=1)
                    plt2.plot(np.arange(0, len(all_offsets)), np.array(all_offsets_estm_1), '-', c='#99C000', lw=.75, label='AUC: %02.2f' % (auc_1*100))
                    plt2.plot(np.arange(0, len(all_offsets)), np.array(all_offsets_estm_2), '-', c='#FDCA00', lw=.75, label='AUC: %02.2f' % (auc_2*100))
                    plt2.set_ylabel('offset')
                    plt2.legend(loc='lower right', fontsize='xx-small')
                    # plt2.set_title('AUC: %2.1f - MSE: %.3f      AUC: %2.1f - MSE: %.3f' % (auc_1*100, MSE1*1000, auc_2*100, MSE2*1000))
                if args.video:
                    # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
                    plt.savefig(os.path.join(png_folder, "frame_%03d.png" % si), dpi=300)
                    plt.savefig(os.path.join(svg_folder, "frame_%03d.svg" % si), dpi=300)
                    writer.grab_frame()


        if args.video:
            plt.close()


        x = np.arange(0, len(all_offsets))
        all_offsets = np.array(all_offsets)
        all_offsets_estm_1 = np.array(all_offsets_estm_1)
        all_offsets_estm_2 = np.array(all_offsets_estm_2)

        plt.figure(figsize=(8, 2.))
        plt.plot(x, all_offsets, '-', c='#000000', lw=.75)
        if checkpoint_path_1 is not None:
            plt.plot(x, all_offsets_estm_1, '-', c='#99C000', lw=.75)
        if checkpoint_path_2 is not None:
            plt.plot(x, all_offsets_estm_2, '-', c='#FDCA00', lw=.75)
        # plt.ylim(-0.25, .25)
        plt.grid(True, which='major', axis='y', color='0.2', linewidth=0.2)
        plt.xlabel('frame')
        plt.ylabel('offset')
        plt.savefig(os.path.join(png_folder, "offsets_%03d.png" % idx), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(svg_folder, "offsets_%03d.svg" % idx), dpi=300, bbox_inches='tight')
        plt.close()


        all_angles = np.array(all_angles)
        all_angles_estm_1 = np.array(all_angles_estm_1)
        all_angles_estm_2 = np.array(all_angles_estm_2)

        plt.figure(figsize=(8, 2.))
        plt.plot(x, all_angles, '-', c='#000000', lw=.5)
        if checkpoint_path_1 is not None:
            plt.plot(x, all_angles_estm_1, '-', c='#99C000', lw=.75)
        if checkpoint_path_2 is not None:
            plt.plot(x, all_angles_estm_2, '-', c='#FDCA00', lw=.75)
        # plt.ylim(-0.1, .1)
        plt.grid(True, which='major', axis='y', color='0.2', linewidth=0.2)
        plt.xlabel('frame')
        plt.ylabel('slope')
        plt.savefig(os.path.join(png_folder, "angles_%03d.png" % idx), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(svg_folder, "angles_%03d.svg" % idx), dpi=300, bbox_inches='tight')
        plt.close()


        all_errors_per_sequence_1 = np.array(all_errors_per_sequence_1)
        all_errors_per_sequence_2 = np.array(all_errors_per_sequence_2)

        plt.figure(figsize=(8, 2.))
        if checkpoint_path_1 is not None:
            plt.plot(x, (all_errors_per_sequence_1), '-', c='#99C000', lw=.75)
        if checkpoint_path_2 is not None:
            plt.plot(x, (all_errors_per_sequence_2), '-', c='#FDCA00', lw=.75)
        # plt.ylim(0, 0.25)
        plt.grid(True, which='major', axis='y', color='0.2', linewidth=0.2)
        plt.xlabel('frame')
        plt.ylabel('error')
        plt.savefig(os.path.join(png_folder, "errors_%03d.png" % idx), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(svg_folder, "errors_%03d.svg" % idx), dpi=300, bbox_inches='tight')
        plt.close()

        if checkpoint_path_1 is not None:
            print("MODEL 1:")
            error_gradient = np.gradient(all_errors_per_sequence_1)
            abs_error_grad = np.sum(np.abs(error_gradient)) / len(all_errors_per_sequence_1)
            MSE = np.mean(np.square(all_errors_per_sequence_1))
            # print(all_errors_per_sequence_1)
            auc, plot_points = calc_auc(np.abs(all_errors_per_sequence_1), cutoff=0.25)
            print("AUC: %.6f" % (auc*100))
            print("MSE: %.8f" % MSE)
            print("Atv: %.8f" % abs_error_grad)

        if checkpoint_path_2 is not None:
            print("MODEL 2:")
            error_gradient = np.gradient(all_errors_per_sequence_2)
            abs_error_grad = np.sum(np.abs(error_gradient)) / len(all_errors_per_sequence_2)
            MSE = np.mean(np.square(all_errors_per_sequence_2))
            auc, plot_points = calc_auc(np.abs(all_errors_per_sequence_2), cutoff=0.25)
            print("AUC: %.6f" % (auc*100))
            print("MSE: %.8f" % MSE)
            print("Atv: %.8f" % abs_error_grad)
        print("")

# print("%d images " % image_count)
#
# mean_err = np.mean(all_errors_1)
# stdd_err = np.std(all_errors_1)
# mean_abserr = np.mean(np.abs(all_errors_1))
# stdd_abserr = np.std(np.abs(all_errors_1))
# max_err = np.max(np.abs(all_errors_1))
#
#
# error_grads_1 = np.concatenate(error_grads_1)
# abs_error_grad = np.sum(np.abs(error_grads_1)) / error_grads_1.shape[0]
# sq_error_grad = np.sum(np.square(np.abs(error_grads_1))) / error_grads_1.shape[0]
# print("abs_error_grad: %.9f" % abs_error_grad)
# print("sq_error_grad: %.9f" % sq_error_grad)
#
# print("total: mean, std, absmean, absstd, max: %.3f %.3f %.3f %.3f %.3f" %
#       (mean_err, stdd_err, mean_abserr, stdd_abserr, max_err))
#
# error_arr = np.abs(np.array(all_errors_1))
# MAE = np.mean(np.abs(error_arr))
# MSE = np.mean(np.square(error_arr))
# print("MSE: %.8f" % MSE)
#
# auc, plot_points = calc_auc(error_arr, cutoff=0.25)
# print("auc: ", auc)
# print("mean error: ", np.mean(error_arr))
#
# plt.figure()
# plt.plot(plot_points[:,0], plot_points[:,1], 'b-')
# plt.xlim(0, 0.25)
# plt.ylim(0, 1.0)
# plt.text(0.175, 0.05, "AUC: %.8f" % auc, fontsize=12)
# plt.suptitle("mean, std, absmean, absstd: %.3f %.3f %.3f %.3f %.3f" %
#              (mean_err, stdd_err, mean_abserr, stdd_abserr, max_err), fontsize=10)
# plt.savefig(os.path.join(png_folder, "error_histogram.png"), dpi=300)
# plt.savefig(os.path.join(svg_folder, "error_histogram.svg"), dpi=300)
#
# print("angular errors:")
# error_arr = np.abs(np.array(all_angular_errors_1))
# auc, plot_points = calc_auc(error_arr, cutoff=5)
# print("auc: ", auc)
# print("mean error: ", np.mean(error_arr))
#
# plt.figure()
# plt.plot(plot_points[:,0], plot_points[:,1], 'b-')
# plt.xlim(0, 5)
# plt.ylim(0, 1.0)
# plt.text(0.175, 0.05, "AUC: %.8f" % auc, fontsize=12)
# plt.suptitle("mean, std, absmean, absstd: %.3f %.3f %.3f %.3f %.3f" %
#              (mean_err, stdd_err, mean_abserr, stdd_abserr, max_err), fontsize=10)
# plt.savefig(os.path.join(png_folder, "error_histogram_angular.png"), dpi=300)
# plt.savefig(os.path.join(svg_folder, "error_histogram_angular.svg"), dpi=300)