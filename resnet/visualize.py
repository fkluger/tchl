import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from resnet.resnet_plus_lstm import resnet18rnn
from datasets.kitti import KittiRawDatasetPP, WIDTH,  HEIGHT
from resnet.train import Config
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

from utilities.gradcam import GradCam, GuidedBackprop
# from utilities.gradcam_misc_functions import

parser = argparse.ArgumentParser(description='')
parser.add_argument('--load', '-l', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--set', '-s', default='val', type=str, metavar='PATH', help='')
parser.add_argument('--gradcam', dest='use_gradcam', action='store_true', help='visualize with gradcam')
parser.add_argument('--guided', dest='guided_gradcam', action='store_true', help='visualize with guided gradcam')
parser.add_argument('--whole', dest='whole_sequence', action='store_true', help='')
parser.add_argument('--video', dest='video', action='store_true', help='')
parser.add_argument('--fchead2', dest='fchead2', action='store_true', help='')
parser.add_argument('--ema', default=0., type=float, metavar='N', help='')
parser.add_argument('--seqlength', default=256, type=int, metavar='N', help='')
parser.add_argument('--cpid', default=-1, type=int, metavar='N', help='')
parser.add_argument('--lstm_mem', default=256, type=int, metavar='N', help='')
parser.add_argument('--disable_mean_subtraction', dest='disable_mean_subtraction', action='store_true', help='visualize with gradcam')
parser.add_argument('--relulstm', dest='relulstm', action='store_true', help='')
parser.add_argument('--skip', dest='skip', action='store_true', help='')

args = parser.parse_args()

args.disable_mean_subtraction = True

checkpoint_path = args.load if not (args.load == '') else None
set_type = args.set

if checkpoint_path is None:
    result_folder = "/home/kluger/tmp/kitti_horizon_videos_2/" + set_type + "/"
else:
    result_folder = os.path.join(checkpoint_path, "results/" + set_type + "/")

if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# device = torch.device('cuda', 0)
device = torch.device('cpu', 0)

seq_length = args.seqlength
whole_sequence = args.whole_sequence

downscale = 2.

if checkpoint_path is not None:
    model = resnet18rnn(load=False, use_fc=True, use_convlstm=True, lstm_mem=args.lstm_mem, second_head=(args.ema > 0),
                        second_head_fc=args.fchead2, relu_lstm=args.relulstm, lstm_skip=args.skip).to(device)
    # model = nn.DataParallel(model)

    if args.cpid == -1:
        cp_path = os.path.join(checkpoint_path, "model_best.ckpt")
    else:
        cp_path_reg = os.path.join(checkpoint_path, "%03d_*.ckpt" % args.cpid)
        cp_path = glob.glob(cp_path_reg)[0]

    print("loading ", cp_path)

    checkpoint = torch.load(cp_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()

if args.use_gradcam:
    gradcam = GradCam(model)
    if args.guided_gradcam:
        gbp = GuidedBackprop(model)
    else:
        gbp = None
else:
    gradcam = None


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
    root_dir = "/data/kluger/datasets/kitti/horizons"
    csv_base = "/home/kluger/tmp/kitti_split_3"
    pdf_file = "/home/kluger/tmp/kitti_split/data_pdfs.pkl"
else:
    target_base = "/data/kluger/checkpoints/horizon_sequences"
    root_dir = "/data/kluger/datasets/kitti/horizons"
    csv_base = "/home/kluger/tmp/kitti_split_3"
    pdf_file = "/home/kluger/tmp/kitti_split/data_pdfs.pkl"


if downscale > 1:
    root_dir += "_s%.3f" % (1./downscale)

if args.ema > 0.:
    root_dir += "_ema%.3f" % args.ema

pixel_mean = [0.362365, 0.377767, 0.366744]
if args.disable_mean_subtraction:
    pixel_mean = [0., 0., 0.]

tfs_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=pixel_mean, std=[1., 1., 1.]),
        ])

im_width = int(WIDTH/downscale)
im_height = int(HEIGHT/downscale)

train_dataset = KittiRawDatasetPP(root_dir=root_dir, pdf_file=pdf_file, augmentation=False, im_height=im_height,
                                  im_width=im_width, return_info=False, get_split_data=(args.ema > 0.),
                                csv_file=csv_base + "/train.csv", seq_length=seq_length, fill_up=False, transform=tfs_val)
val_dataset = KittiRawDatasetPP(root_dir=root_dir, pdf_file=pdf_file, augmentation=False, im_height=im_height,
                                im_width=im_width, return_info=False, get_split_data=(args.ema > 0.),
                              csv_file=csv_base + "/val.csv", seq_length=seq_length, fill_up=False, transform=tfs_val)
test_dataset = KittiRawDatasetPP(root_dir=root_dir, pdf_file=pdf_file, augmentation=False, im_height=im_height,
                                 im_width=im_width, return_info=False, get_split_data=(args.ema > 0.),
                              csv_file=csv_base + "/test.csv", seq_length=seq_length, fill_up=False, transform=tfs_val)

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

result_folder = os.path.join(result_folder, "%d/" % args.lstm_mem)

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

with (torch.enable_grad() if args.use_gradcam else torch.no_grad()):
    losses = []
    offset_losses = []
    angle_losses = []
    for idx, sample in enumerate(loader):

        images = sample['images']
        offsets = sample['offsets']
        angles = sample['angles']
        if args.ema > 0:
            offsets_ema = sample['offsets_ema']
            angles_ema = sample['angles_ema']
            offsets_dif = offsets - offsets_ema
            angles_dif = angles - angles_ema

        # K = sample['K']
        # Gs = sample['G']

        print("idx ", idx)
        all_offsets = []
        all_offsets_ema = []
        all_offsets_dif = []
        all_offsets_estm = []
        all_offsets_ema_estm = []
        all_offsets_dif_estm = []
        all_angles = []
        all_angles_ema = []
        all_angles_dif = []
        all_angles_estm = []
        all_angles_ema_estm = []
        all_angles_dif_estm = []

        if args.video:
            fig = plt.figure(figsize=(6.4, 3.0))

            l1, = plt.plot([], [], '-', lw=2, c='#99C000')
            l2, = plt.plot([], [], '--', lw=2, c='#0083CC')
            if args.ema > 0:
                l3, = plt.plot([], [], '--', lw=1.2, c='#fdca00')
                # l4, = plt.plot([], [], '--', lw=1.2, c='#fdca00')


        with writer.saving(fig, video_folder + "%s%05d.mp4" % (("guided-" if args.guided_gradcam else "") +
                                                               ("gradcam_" if args.use_gradcam else ""), idx), 300) \
                if args.video else contextlib.suppress():
            if whole_sequence and checkpoint_path is not None:
                if args.ema > 0:
                    output_offsets_dif, output_angles_dif, output_offsets_ema, output_angles_ema = model(images.to(device))
                    output_offsets = output_offsets_ema + output_offsets_dif
                    output_angles = output_angles_ema + output_angles_dif
                else:
                    output_offsets, output_angles = model(images.to(device))
            for si in range(images.shape[1]):

                image = images.numpy()[0,si,:,:,:].transpose((1,2,0))
                width = image.shape[1]
                height = image.shape[0]

                offset = offsets[0, si].detach().numpy().squeeze()
                angle = angles[0, si].detach().numpy().squeeze()

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

                if args.ema > 0:
                    offset_ema = offsets_ema[0, si].detach().numpy().squeeze()
                    angle_ema = angles_ema[0, si].detach().numpy().squeeze()

                    all_offsets_ema += [-offset_ema.copy()]
                    all_angles_ema += [angle_ema.copy()]

                    offset_ema += 0.5
                    offset_ema *= height

                    true_mp_ema = np.array([width/2., offset_ema])
                    true_nv_ema = np.array([np.sin(angle_ema), np.cos(angle_ema)])
                    true_hl_ema = np.array([true_nv_ema[0], true_nv_ema[1], -np.dot(true_nv_ema, true_mp_ema)])
                    true_h1_ema = np.cross(true_hl_ema, np.array([1, 0, 0]))
                    true_h2_ema = np.cross(true_hl_ema, np.array([1, 0, -width]))
                    true_h1_ema /= true_h1_ema[2]
                    true_h2_ema /= true_h2_ema[2]
                    
                    offset_dif = offsets_dif[0, si].detach().numpy().squeeze()
                    angle_dif = angles_dif[0, si].detach().numpy().squeeze()

                    all_offsets_dif += [-offset_dif.copy()]
                    all_angles_dif += [angle_dif.copy()]

                    offset_dif += 0.5
                    offset_dif *= height

                    true_mp_dif = np.array([width/2., offset_dif])
                    true_nv_dif = np.array([np.sin(angle_dif), np.cos(angle_dif)])
                    true_hl_dif = np.array([true_nv_dif[0], true_nv_dif[1], -np.dot(true_nv_dif, true_mp_dif)])
                    true_h1_dif = np.cross(true_hl_dif, np.array([1, 0, 0]))
                    true_h2_dif = np.cross(true_hl_dif, np.array([1, 0, -width]))
                    true_h1_dif /= true_h1_dif[2]
                    true_h2_dif /= true_h2_dif[2]

                plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
                    # fig.set_dpi(plt.rcParams["figure.dpi"])

                if checkpoint_path is not None:
                    if not whole_sequence:
                        if args.ema > 0:
                            output_offsets_dif, output_angles_dif, output_offsets_ema, output_angles_ema = model(images[:,si,:,:,:].unsqueeze(1).to(device))
                            output_offsets = output_offsets_ema + output_offsets_dif
                            output_angles = output_angles_ema + output_angles_dif
                            offset_ema_estm = output_offsets_ema[0,0].cpu().detach().numpy().squeeze()
                            angle_ema_estm = output_angles_ema[0,0].cpu().detach().numpy().squeeze()
                            all_offsets_ema_estm += [-offset_ema_estm.copy()]
                            all_angles_ema_estm += [angle_ema_estm.copy()]
                            offset_dif_estm = output_offsets_dif[0,0].cpu().detach().numpy().squeeze()
                            angle_dif_estm = output_angles_dif[0,0].cpu().detach().numpy().squeeze()
                            all_offsets_dif_estm += [-offset_dif_estm.copy()]
                            all_angles_dif_estm += [angle_dif_estm.copy()]
                        else:
                            output_offsets, output_angles = model(images[:,si,:,:,:].unsqueeze(1).to(device))
                        offset_estm = output_offsets[0,0].cpu().detach().numpy().squeeze()
                        angle_estm = output_angles[0,0].cpu().detach().numpy().squeeze()

                    else:
                        offset_estm = output_offsets[0,si].cpu().detach().numpy().squeeze()
                        angle_estm = output_angles[0,si].cpu().detach().numpy().squeeze()
                        if args.ema > 0:
                            offset_ema_estm = output_offsets_ema[0,si].cpu().detach().numpy().squeeze()
                            angle_ema_estm = output_angles_ema[0,si].cpu().detach().numpy().squeeze()
                            all_offsets_ema_estm += [-offset_ema_estm.copy()]
                            all_angles_ema_estm += [angle_ema_estm.copy()]
                            offset_dif_estm = output_offsets_dif[0,si].cpu().detach().numpy().squeeze()
                            angle_dif_estm = output_angles_dif[0,si].cpu().detach().numpy().squeeze()
                            all_offsets_dif_estm += [-offset_dif_estm.copy()]
                            all_angles_dif_estm += [angle_dif_estm.copy()]

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

                    # G = Gs.cpu().numpy().squeeze()[si,:]
                    # print("G: ", G.shape)

                    # ax3d.plot([G[0]], [G[1]], [G[2]], 'b.')

                    if gradcam is not None:
                        img_as_var = torch.autograd.Variable(images[0, si, :, :, :].clone().unsqueeze(0).unsqueeze(0),
                                                             requires_grad=True)
                        # target = np.array([np.sign(offset-offset_estm)])
                        cam_up, target_up = gradcam.generate_cam(images[0, si, :, :, :], np.array([1]))
                        cam_down, target_down = gradcam.generate_cam(images[0, si, :, :, :], np.array([-1]))

                        cam_up = cv2.resize(cam_up, (WIDTH, HEIGHT),
                                            interpolation=cv2.INTER_NEAREST)
                        cam_down = cv2.resize(cam_down, (WIDTH, HEIGHT),
                                              interpolation=cv2.INTER_NEAREST)
                        if gbp is not None:
                            guided_grads = gbp.generate_gradients(img_as_var, target)
                            grayscale_guided_grads = convert_to_grayscale(guided_grads)
                            # gradient_img = gradient_images(guided_grads)
                            gray_gradient_img = gradient_images(grayscale_guided_grads).squeeze()
                        del img_as_var


                # print("true: %.2f px, %.2f deg" % (offset, angle*180./np.pi))
                # print("estm: %.2f px, %.2f deg" % (offset_estm, angle_estm*180./np.pi))

                image[:,:,0] += pixel_mean[0]
                image[:,:,1] += pixel_mean[1]
                image[:,:,2] += pixel_mean[2]

                if args.use_gradcam:
                    if cam_up is not None:
                        if gbp is not None:
                            image = class_activation_on_image(image*255., gray_gradient_img)
                        else:
                            image = class_activation_on_image_combined(image*255., cam_up.astype(np.float32)-cam_down.astype(np.float32))

                if args.video:
                    plt.imshow(image)
                    plt.axis('off')
                    plt.autoscale(False)

                    l1.set_data([true_h1[0], true_h2[0]], [true_h1[1], true_h2[1]])

                    if args.ema > 0:
                        l3.set_data([true_h1_ema[0], true_h2_ema[0]], [true_h1_ema[1], true_h2_ema[1]])
                        # l4.set_data([true_h1_dif[0], true_h2_dif[0]], [true_h1_dif[1], true_h2_dif[1]])

                    if checkpoint_path is not None:
                        l2.set_data([estm_h1[0], estm_h2[0]], [estm_h1[1], estm_h2[1]])

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

        plt.figure()
        x = np.arange(0, len(all_offsets))
        all_offsets = np.array(all_offsets)
        all_offsets_ema = np.array(all_offsets_ema)
        all_offsets_dif = np.array(all_offsets_dif)
        all_offsets_estm = np.array(all_offsets_estm)
        all_offsets_ema_estm = np.array(all_offsets_ema_estm)
        all_offsets_dif_estm = np.array(all_offsets_dif_estm)
        # print(all_offsets)
        plt.plot(x, all_offsets, '-', c='#99C000')
        if args.ema > 0:
            plt.plot(x, all_offsets_ema, '-', c='#fdca00')
            plt.plot(x, all_offsets_dif, '-', c='#fdca00')
        if checkpoint_path is not None:
            plt.plot(x, all_offsets_estm, '-', c='#0083CC')
            if args.ema > 0:
                plt.plot(x, all_offsets_ema_estm, '-', c='#A60084')
                plt.plot(x, all_offsets_dif_estm, '-', c='#A60084')
        plt.ylim(-.4, .4)

        if checkpoint_path is not None:
            errors = np.abs(all_offsets-all_offsets_estm)
            err_mean = np.mean(errors).squeeze()
            err_stdd = np.std(errors).squeeze()
            plt.suptitle("error mean: %.4f -- stdd: %.4f" % (err_mean, err_stdd))

        plt.savefig(os.path.join(png_folder, "offsets_%03d.png" % idx), dpi=300)
        plt.savefig(os.path.join(svg_folder, "offsets_%03d.svg" % idx), dpi=300)
        plt.close()

        plt.figure()
        x = np.arange(0, len(all_angles))
        all_angles = np.array(all_angles)
        all_angles_ema = np.array(all_angles_ema)
        all_angles_dif = np.array(all_angles_dif)
        all_angles_estm = np.array(all_angles_estm)
        all_angles_ema_estm = np.array(all_angles_ema_estm)
        all_angles_dif_estm = np.array(all_angles_dif_estm)
        # print(all_offsets)
        plt.plot(x, all_angles, '-', c='#99C000')
        if args.ema > 0:
            plt.plot(x, all_angles_ema, '-', c='#fdca00')
            plt.plot(x, all_angles_dif, '-', c='#fdca00')
        if checkpoint_path is not None:
            plt.plot(x, all_angles_estm, '-', c='#0083CC')
            if args.ema > 0:
                plt.plot(x, all_angles_ema_estm, '-', c='#A60084')
                plt.plot(x, all_angles_dif_estm, '-', c='#A60084')
        plt.ylim(-.4, .4)

        if checkpoint_path is not None:
            errors = np.abs(all_angles-all_angles_estm)
            err_mean = np.mean(errors).squeeze()
            err_stdd = np.std(errors).squeeze()
            plt.suptitle("error mean: %.4f -- stdd: %.4f" % (err_mean, err_stdd))

        plt.savefig(os.path.join(png_folder, "angles_%03d.png" % idx), dpi=300)
        plt.savefig(os.path.join(svg_folder, "angles_%03d.svg" % idx), dpi=300)
        plt.close()

