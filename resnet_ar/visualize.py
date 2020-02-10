import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from resnet import resnet_plus_lstm
from resnet_3d import resnet_3d_models
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

from utilities.gradcam import GradCam, GuidedBackprop
# from utilities.gradcam_misc_functions import

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
parser.add_argument('--load', '-l', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--set', '-s', default='val', type=str, metavar='PATH', help='')
parser.add_argument('--whole', dest='whole_sequence', action='store_true', help='')
parser.add_argument('--video', dest='video', action='store_true', help='')
parser.add_argument('--seqlength', default=512, type=int, metavar='N', help='')
parser.add_argument('--cpid', default=-1, type=int, metavar='N', help='')
parser.add_argument('--disable_mean_subtraction', dest='disable_mean_subtraction', action='store_true', help='visualize with gradcam')
parser.add_argument('--cpu', dest='cpu', action='store_true', help='')
parser.add_argument('--gpu', default='0', type=str, metavar='DS', help='dataset')
parser.add_argument('--net', default='resnet18_3_2d_1_3d', type=str, metavar='DS', help='dataset')
parser.add_argument('--bias', dest='bias', action='store_true', help='')
parser.add_argument('--features', dest='features', action='store_true', help='')
parser.add_argument('--tee', dest='tee', action='store_true', help='')

args = parser.parse_args()

checkpoint_path = args.load if not (args.load == '') else None
set_type = args.set

if checkpoint_path is None:
    result_folder = "/home/kluger/tmp/kitti_horizon_videos_2/" + set_type + "/"
else:
    result_folder = os.path.join(checkpoint_path, "results/" + set_type + "/")

if not os.path.exists(result_folder):
    os.makedirs(result_folder)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if args.cpu:
    device = torch.device('cpu', 0)
else:
    device = torch.device('cuda', 0)
# device = torch.device('cpu', 0)

seq_length = args.seqlength
whole_sequence = args.whole_sequence

downscale = 2.

if checkpoint_path is not None:

    if args.net == 'res18':
        modelfun = resnet_3d_models.resnet18_3d_3_3
    elif args.net == 'resnet18_2d3d_3_3dil':
        modelfun = resnet_3d_models.resnet18_2d3d_3_3dil
    elif args.net == 'resnet18_2d':
        modelfun = resnet_3d_models.resnet18_2d
    elif args.net == 'resnet18':
        modelfun = resnet_3d_models.resnet18
    elif args.net == 'resnet18_3_2d_1_3d':
        modelfun = resnet_3d_models.resnet18_3_2d_1_3d
    elif args.net == 'resnet18_2_2d_2_3d':
        modelfun = resnet_3d_models.resnet18_2_2d_2_3d
    elif args.net == 'resnet18rnn':
        modelfun = resnet_plus_lstm.resnet18rnn
    else:
        assert False

    model, blocks = modelfun(order='BDCHW')
    model = model.to(device)

    fov_increase = model.fov_increase
    overlap = 2*fov_increase

    if args.cpid == -1:
        cp_path = os.path.join(checkpoint_path, "model_best.ckpt")
    else:
        cp_path_reg = os.path.join(checkpoint_path, "%03d_*.ckpt" % args.cpid)
        cp_path = glob.glob(cp_path_reg)[0]

    load_from_path = cp_path
    print("load weights from ", load_from_path)
    checkpoint = torch.load(load_from_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()

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
elif 'hekate' in hostname:
    target_base = "/data/kluger/checkpoints/horizon_sequences"
    root_dir = "/phys/ssd/kitti/horizons"
    csv_base = "/home/kluger/tmp/kitti_split_3"
    pdf_file = "/home/kluger/tmp/kitti_split/data_pdfs.pkl"
else:
    target_base = "/data/kluger/checkpoints/horizon_sequences"
    root_dir = "/data/kluger/datasets/kitti/horizons"
    csv_base = "/home/kluger/tmp/kitti_split_3"
    pdf_file = "/home/kluger/tmp/kitti_split/data_pdfs.pkl"


if downscale > 1:
    root_dir += "_s%.3f" % (1./downscale)


root_dir += "_ema%.3f" % 0.1


pixel_mean = [0.362365, 0.377767, 0.366744]
if args.disable_mean_subtraction:
    pixel_mean = [0., 0., 0.]

# print("pixel_mean: ", pixel_mean)

tfs_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=pixel_mean, std=[1., 1., 1.]),
        ])

im_width = int(WIDTH/downscale)
im_height = int(HEIGHT/downscale)

calc_hlr = calc_horizon_leftright(im_width, im_height)

train_dataset = KittiRawDatasetPP(root_dir=root_dir, pdf_file=pdf_file, augmentation=False, im_height=im_height,
                                  im_width=im_width, return_info=True, get_split_data=False,
                                  csv_file=csv_base + "/train.csv", seq_length=seq_length, fill_up=False,
                                  transform=tfs_val, pre_padding=overlap, overlap=overlap)
val_dataset = KittiRawDatasetPP(root_dir=root_dir, pdf_file=pdf_file, augmentation=False, im_height=im_height,
                                im_width=im_width, return_info=True, get_split_data=False,
                                csv_file=csv_base + "/val.csv", seq_length=seq_length, fill_up=False,
                                transform=tfs_val, pre_padding=overlap, overlap=overlap)
test_dataset = KittiRawDatasetPP(root_dir=root_dir, pdf_file=pdf_file, augmentation=False, im_height=im_height,
                                 im_width=im_width, return_info=True, get_split_data=False,
                                 csv_file=csv_base + "/test.csv", seq_length=seq_length, fill_up=False,
                                 transform=tfs_val, pre_padding=overlap, overlap=overlap)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=1,
                                           shuffle=False)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                          batch_size=1,
                                          shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=False)

if set_type == 'val':
    loader = val_loader
elif set_type == 'test':
    loader = test_loader
else:
    loader = train_loader

result_folder = os.path.join(result_folder, "%d/" % args.seqlength)


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

if args.tee:
    log_file = os.path.join(result_folder, "log")
    log = Tee(os.path.join(result_folder, log_file), "w", file_only=False)

all_errors = []
all_angular_errors = []

image_count = 0

percs = [50, 80, 90, 95, 99]
print("percentiles: ", percs)

with torch.no_grad():
    losses = []
    offset_losses = []
    angle_losses = []
    for idx, sample in enumerate(loader):

        images = sample['images']
        offsets = sample['offsets']
        angles = sample['angles']
        Gs = sample['G'][0]
        padding = sample['padding']
        scale = sample['scale'].numpy()
        K = np.matrix(sample['K'][0])
        # K = sample['K']
        # Gs = sample['G']

        print("idx ", idx, end="\t")
        all_offsets = []
        all_offsets_estm = []
        all_angles = []
        all_angles_estm = []
        feature_similarities = []
        feat = None

        all_errors_per_sequence = []
        all_angular_errors_per_sequence = []

        if args.video:
            fig = plt.figure(figsize=(6.4, 3.0))

            l1, = plt.plot([], [], '-', lw=2, c='#99C000')
            l2, = plt.plot([], [], '--', lw=2, c='#0083CC')


        with writer.saving(fig, video_folder + "%s%05d.mp4" % (("guided-" if args.guided_gradcam else "") +
                                                               ("gradcam_" if args.use_gradcam else ""), idx), 300) \
                                                                if args.video else contextlib.suppress():
            if whole_sequence and checkpoint_path is not None:

                if args.features:
                    output_offsets, output_angles, feat = model(images.to(device), get_features=True)
                else:
                    output_offsets, output_angles = model(images.to(device))

                # print(output_offsets[0,:8])
                # exit(0)
            for si in range(images.shape[1]-2*fov_increase):

                image_count += 1

                image = images.numpy()[0,si+2*fov_increase,:,:,:].transpose((1,2,0))
                width = image.shape[1]
                height = image.shape[0]
                # print(image.shape)
                # print(image[20,100:108,0])
                # exit(0)

                offset = offsets[0, si].detach().numpy().squeeze()
                angle = angles[0, si].detach().numpy().squeeze()

                if args.features:
                    if si > 0 and feat is not None:
                        feat1 = feat[0, si-1, :].cpu().detach().numpy().squeeze()
                        feat2 = feat[0, si, :].cpu().detach().numpy().squeeze()
                        similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1)*np.linalg.norm(feat2))
                        feature_similarities += [similarity]

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

                plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
                    # fig.set_dpi(plt.rcParams["figure.dpi"])

                if checkpoint_path is not None:
                    if not whole_sequence:
                        output_offsets, output_angles = model(images[:,si,:,:,:].unsqueeze(1).to(device))
                        offset_estm = output_offsets[0,fov_increase].cpu().detach().numpy().squeeze()
                        angle_estm = output_angles[0,fov_increase].cpu().detach().numpy().squeeze()

                        yle, yre = calc_hlr(output_offsets[0, fov_increase], output_angles[0, fov_increase])
                    else:
                        offset_estm = output_offsets[0,si+fov_increase].cpu().detach().numpy().squeeze()
                        angle_estm = output_angles[0,si+fov_increase].cpu().detach().numpy().squeeze()

                        yle, yre = calc_hlr(output_offsets[0, si+fov_increase], output_angles[0, si+fov_increase])


                    all_offsets_estm += [-offset_estm.copy()]
                    all_angles_estm += [angle_estm.copy()]

                    offset_estm += 0.5
                    offset_estm *= height

                    estm_mp = np.array([width/2., offset_estm])
                    estm_nv = np.array([np.sin(angle_estm), np.cos(angle_estm)])
                    estm_hl = np.array([estm_nv[0], estm_nv[1], -np.dot(estm_nv, estm_mp)])
                    estm_h1 = np.cross(estm_hl, np.array([1, 0, 0]))
                    estm_h2 = np.cross(estm_hl, np.array([1, 0, -width]))
                    estm_h1 /= estm_h1[2]
                    estm_h2 /= estm_h2[2]

                    h1_ = estm_h1 / scale - np.array([padding[0], padding[2], 1 / scale - 1])
                    h2_ = estm_h2 / scale - np.array([padding[0], padding[2], 1 / scale - 1])
                    h_ = np.cross(h1_, h2_)
                    Ge = K.T * np.matrix(h_).T
                    Ge /= np.linalg.norm(Ge)

                    G = np.matrix(Gs[si]).T
                    G /= np.linalg.norm(G)

                    err1 = (yl-yle) / height
                    err2 = (yr-yre) / height

                    # err1 = np.minimum(np.linalg.norm(estm_h1 - true_h1), np.linalg.norm(estm_h1 - true_h2))
                    # err2 = np.minimum(np.linalg.norm(estm_h2 - true_h1), np.linalg.norm(estm_h2 - true_h2))

                    err = np.maximum(err1, err2) / height

                    if np.abs(err1) > np.abs(err2):
                        err = err1
                    else:
                        err = err2

                    all_errors.append(err)
                    all_errors_per_sequence.append(err)

                    try:
                        angular_error = np.abs((np.arccos(np.clip(np.abs(np.dot(Ge.T, G)), 0, 1))*180/np.pi)[0,0])
                    except:
                        angular_error = 0
                        print(Ge)
                        print(G)
                        print(np.dot(Ge.T, G))
                        exit(0)

                    all_angular_errors.append(angular_error)
                    all_angular_errors_per_sequence.append(angular_error)

                if args.video:
                    image[:,:,0] += pixel_mean[0]
                    image[:,:,1] += pixel_mean[1]
                    image[:,:,2] += pixel_mean[2]


                if args.video:
                    plt.imshow(image)
                    plt.axis('off')
                    plt.autoscale(False)

                    l1.set_data([true_h1[0], true_h2[0]], [true_h1[1], true_h2[1]])
                    # l1.set_data([0, yl], [im_width-1, yr])

                    if checkpoint_path is not None:
                        l2.set_data([estm_h1[0], estm_h2[0]], [estm_h1[1], estm_h2[1]])
                        # l2.set_data([0, yle], [im_width-1, yre])

                        plt.suptitle("true: %.1f px, %.1f deg --- error: %.1f px, %.1f deg" %
                                     (offset, angle*180./np.pi, np.abs(offset-offset_estm),
                                      np.abs(angle-angle_estm)*180./np.pi), family='monospace', y=0.9)
                    else:
                        plt.suptitle("%.1f px, %.1f deg" %
                                     (offset, angle*180./np.pi), family='monospace', y=0.9)

                    # plt.show()
                    writer.grab_frame()
                    # plt.clf()


        if args.video:
            plt.close()

        mean_err = np.mean(all_errors_per_sequence)
        stdd_err = np.std(all_errors_per_sequence)
        mean_abserr = np.mean(np.abs(all_errors_per_sequence))
        stdd_abserr = np.std(np.abs(all_errors_per_sequence))
        max_err = np.max(np.abs(all_errors_per_sequence))

        perc_values = np.percentile(np.abs(all_errors_per_sequence), percs)
        # print(percs)
        for pv in perc_values: print("%.3f " % pv, end="")
        perc_values = np.percentile(np.abs(all_angular_errors_per_sequence), percs)
        print(" | ")
        for pv in perc_values: print("%.3f " % pv, end="")
        print("")

        plt.figure()
        x = np.arange(0, len(all_offsets))
        all_offsets = np.array(all_offsets)
        all_offsets_estm = np.array(all_offsets_estm)

        corr = np.correlate(all_offsets, all_offsets_estm, "same")
        max_corr_off = np.max(corr)

        # print("mean, std, absmean, absstd, max: %.3f %.3f %.3f %.3f %.3f | corr: %.3f" %
        #       (mean_err, stdd_err, mean_abserr, stdd_abserr, max_err, max_corr_off))

        # print(all_offsets)
        plt.plot(x, all_offsets, '-', c='#99C000')
        if checkpoint_path is not None:
            plt.plot(x, all_offsets_estm, '-', c='#0083CC')
        plt.ylim(-.4, .4)

        if checkpoint_path is not None:
            errors = np.abs(all_offsets-all_offsets_estm)
            err_mean = np.mean(errors).squeeze()
            err_stdd = np.std(errors).squeeze()
            # plt.suptitle("error mean: %.4f -- stdd: %.4f" % (err_mean, err_stdd))
            plt.suptitle("mean: %.4f - stdd: %.4f | mean, std, absmean, absstd: %.3f %.3f %.3f %.3f %.3f | corr: %.3f" %
                         (err_mean, err_stdd, mean_err, stdd_err, mean_abserr, stdd_abserr, max_err, max_corr_off), fontsize=8)


        plt.savefig(os.path.join(png_folder, "offsets_%03d.png" % idx), dpi=300)
        plt.savefig(os.path.join(svg_folder, "offsets_%03d.svg" % idx), dpi=300)
        plt.close()

        plt.figure()
        x = np.arange(0, len(all_angles))
        all_angles = np.array(all_angles)
        all_angles_estm = np.array(all_angles_estm)

        corr = np.correlate(all_angles, all_angles_estm, "same")
        max_corr_ang = np.max(corr)
        print("mean, std, absmean, absstd, max: %.3f %.3f %.3f %.3f %.3f | corr: %.3f, %.3f" %
              (mean_err, stdd_err, mean_abserr, stdd_abserr, max_err, max_corr_off, max_corr_ang))

        # print(all_offsets)
        plt.plot(x, all_angles, '-', c='#99C000')
        if checkpoint_path is not None:
            plt.plot(x, all_angles_estm, '-', c='#0083CC')
        plt.ylim(-.4, .4)

        if checkpoint_path is not None:
            errors = np.abs(all_angles-all_angles_estm)
            err_mean = np.mean(errors).squeeze()
            err_stdd = np.std(errors).squeeze()
            # plt.suptitle("error mean: %.4f -- stdd: %.4f" % (err_mean, err_stdd))
            plt.suptitle("mean: %.4f - stdd: %.4f | mean, std, absmean, absstd: %.3f %.3f %.3f %.3f %.3f | corr: %.3f" %
                         (err_mean, err_stdd, mean_err, stdd_err, mean_abserr, stdd_abserr, max_err, max_corr_ang), fontsize=8)

        plt.savefig(os.path.join(png_folder, "angles_%03d.png" % idx), dpi=300)
        plt.savefig(os.path.join(svg_folder, "angles_%03d.svg" % idx), dpi=300)
        plt.close()

        if args.features:
            if feat is not None:
                plt.figure()
                x = np.arange(1, len(feature_similarities)+1)
                feature_similarities = np.array(feature_similarities)
                # print(all_offsets)
                plt.plot(x, feature_similarities, '-', c='#99C000')
                plt.ylim(-1, 1)

                plt.savefig(os.path.join(png_folder, "features_%03d.png" % idx), dpi=300)
                plt.savefig(os.path.join(svg_folder, "features_%03d.svg" % idx), dpi=300)
                plt.close()

print("%d images " % image_count)

mean_err = np.mean(all_errors)
stdd_err = np.std(all_errors)
mean_abserr = np.mean(np.abs(all_errors))
stdd_abserr = np.std(np.abs(all_errors))
max_err = np.max(np.abs(all_errors))

perc_values = np.percentile(np.abs(all_errors), percs)
for pv in perc_values: print("%.3f " % pv, end="")
print(" | ")
perc_values = np.percentile(np.abs(all_angular_errors), percs)
for pv in perc_values: print("%.3f " % pv, end="")
print("")

print("total: mean, std, absmean, absstd, max: %.3f %.3f %.3f %.3f %.3f" %
      (mean_err, stdd_err, mean_abserr, stdd_abserr, max_err))


error_arr = np.abs(np.array(all_errors))
auc, plot_points = calc_auc(error_arr, cutoff=0.25)
print("auc: ", auc)
print("mean error: ", np.mean(error_arr))

plt.figure()
plt.plot(plot_points[:,0], plot_points[:,1], 'b-')
plt.xlim(0, 0.25)
plt.ylim(0, 1.0)
plt.text(0.175, 0.05, "AUC: %.8f" % auc, fontsize=12)
plt.suptitle("mean, std, absmean, absstd: %.3f %.3f %.3f %.3f %.3f" %
             (mean_err, stdd_err, mean_abserr, stdd_abserr, max_err), fontsize=10)
plt.savefig(os.path.join(png_folder, "error_histogram.png"), dpi=300)
plt.savefig(os.path.join(svg_folder, "error_histogram.svg"), dpi=300)

print("angular errors:")
error_arr = np.abs(np.array(all_angular_errors))
auc, plot_points = calc_auc(error_arr, cutoff=5)
print("auc: ", auc)
print("mean error: ", np.mean(error_arr))

plt.figure()
plt.plot(plot_points[:,0], plot_points[:,1], 'b-')
plt.xlim(0, 5)
plt.ylim(0, 1.0)
plt.text(0.175, 0.05, "AUC: %.8f" % auc, fontsize=12)
plt.suptitle("mean, std, absmean, absstd: %.3f %.3f %.3f %.3f %.3f" %
             (mean_err, stdd_err, mean_abserr, stdd_abserr, max_err), fontsize=10)
plt.savefig(os.path.join(png_folder, "error_histogram_angular.png"), dpi=300)
plt.savefig(os.path.join(svg_folder, "error_histogram_angular.svg"), dpi=300)
