import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from resnet.resnet_plus_lstm import resnet18rnn
from datasets.kitti import KittiRawDatasetPP, WIDTH,  HEIGHT
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
import matplotlib.animation as manimation
import platform
import logging
logger = logging.getLogger('matplotlib.animation')
logger.setLevel(logging.DEBUG)
hostname = platform.node()
import argparse

from utilities.gradcam import GradCam, GuidedBackprop
# from utilities.gradcam_misc_functions import

parser = argparse.ArgumentParser(description='')
parser.add_argument('--load', '-l', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--set', '-s', default='val', type=str, metavar='PATH', help='')
parser.add_argument('--gradcam', dest='use_gradcam', action='store_true', help='visualize with gradcam')
parser.add_argument('--guided', dest='guided_gradcam', action='store_true', help='visualize with guided gradcam')
parser.add_argument('--whole', dest='whole_sequence', action='store_true', help='')
parser.add_argument('--seqlength', default=256, type=int, metavar='N', help='')
parser.add_argument('--disable_mean_subtraction', dest='disable_mean_subtraction', action='store_true', help='visualize with gradcam')

args = parser.parse_args()

args.disable_mean_subtraction = True

# checkpoint_path = "/data/kluger/checkpoints/horizon_sequences/res18_fine/1/b32_180910-133634/009_0.001100.ckpt"
# checkpoint_path = "/data/kluger/checkpoints/horizon_sequences/res18_fine/16/b2_180911-104511/007_0.001957.ckpt"
# checkpoint_path = "/data/kluger/checkpoints/horizon_sequences/res18_fine/1/b2_180913-104954/051_0.000588.ckpt"
# checkpoint_path = "/tnt/data/kluger/checkpoints/horizon_sequences/res18_fine/1/b2_180914-104717/006_0.000127.ckpt"
# checkpoint_path = "/data/kluger/checkpoints/horizon_sequences/res18_fine/1/b2_180914-143448/003_0.001422.ckpt"
# checkpoint_path = "/data/kluger/checkpoints/horizon_sequences/res18_fine/1/b2_180915-184354/017_0.001131.ckpt"
# checkpoint_path = "/data/kluger/checkpoints/horizon_sequences/res18_fine/2/b2_180916-101747/model_best.ckpt"
# checkpoint_path = "/data/kluger/checkpoints/horizon_sequences/res18_fine/d1/1/b2_180919-215524/"
# checkpoint_path = "/data/kluger/checkpoints/horizon_sequences/res18_fine/d1/4/b2_180920-172021/"
# checkpoint_path = "/data/kluger/checkpoints/horizon_sequences/res18_fine/d1/1/b2_180921-223047/"

checkpoint_path = args.load if not (args.load == '') else None
set_type = args.set

if checkpoint_path is None:
    result_folder = "/home/kluger/tmp/kitti_horizon_videos/" + set_type + "/"
else:
    result_folder = os.path.join(checkpoint_path, "results/" + set_type + "/")

if not os.path.exists(result_folder):
    os.makedirs(result_folder)

device = torch.device('cuda', 0)

seq_length = args.seqlength
whole_sequence = args.whole_sequence

downscale = 2.

if checkpoint_path is not None:
    model = resnet18rnn(use_fc=True, use_convlstm=True).to(device)
    checkpoint = torch.load(os.path.join(checkpoint_path, "model_best.ckpt"), map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
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
else:
    target_base = "/data/kluger/checkpoints/horizon_sequences"
    root_dir = "/data/kluger/datasets/kitti/horizons"
    csv_base = "/home/kluger/tmp/kitti_split_3"
    pdf_file = "/home/kluger/tmp/kitti_split/data_pdfs.pkl"


if downscale > 1:
    root_dir += "_s%.3f" % (1./downscale)

pixel_mean = [0.362365, 0.377767, 0.366744]
if args.disable_mean_subtraction:
    pixel_mean = [0., 0., 0.]

tfs_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=pixel_mean, std=[1., 1., 1.]),
        ])

im_width = int(WIDTH/downscale)
im_height = int(HEIGHT/downscale)

train_dataset = KittiRawDatasetPP(root_dir=root_dir, pdf_file=pdf_file, augmentation=False, im_height=im_height, im_width=im_width,
                                csv_file=csv_base + "/train.csv", seq_length=seq_length, fill_up=False, transform=tfs_val)
val_dataset = KittiRawDatasetPP(root_dir=root_dir, pdf_file=pdf_file, augmentation=False, im_height=im_height, im_width=im_width,
                              csv_file=csv_base + "/val.csv", seq_length=seq_length, fill_up=False, transform=tfs_val)
test_dataset = KittiRawDatasetPP(root_dir=root_dir, pdf_file=pdf_file, augmentation=False, im_height=im_height, im_width=im_width,
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
    result_folder = os.path.join(result_folder, "val/")
elif set_type == 'test':
    loader = test_loader
    result_folder = os.path.join(result_folder, "test/")
else:
    loader = train_loader
    result_folder = os.path.join(result_folder, "train/")

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

        # if idx < 49: continue

        images = sample['images']
        offsets = sample['offsets']
        angles = sample['angles']

        print("idx ", idx)
        all_offsets = []
        all_offsets_estm = []

        fig = plt.figure(figsize=(6.4, 3.0))
        # print("figsize: ", plt.rcParams['figure.figsize'])

        l1, = plt.plot([], [], '-', lw=2, c='#99C000')
        l2, = plt.plot([], [], '--', lw=2, c='#0083CC')

        with writer.saving(fig, video_folder + "%s%05d.mp4" % (("guided-" if args.guided_gradcam else "") + ("gradcam_" if args.use_gradcam else ""), idx), 300):
            if whole_sequence and checkpoint_path is not None:
                output_offsets, output_angles = model(images.to(device))
            for si in range(images.shape[1]):

                image = images.numpy()[0,si,:,:,:].transpose((1,2,0))
                width = image.shape[1]
                height = image.shape[0]

                offset = offsets[0, si].detach().numpy().squeeze()
                angle = angles[0, si].detach().numpy().squeeze()

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

                fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
                    # fig.set_dpi(plt.rcParams["figure.dpi"])

                if checkpoint_path is not None:
                    if not whole_sequence:
                        output_offsets, output_angles = model(images[:,si,:,:,:].unsqueeze(1).to(device))
                        offset_estm = output_offsets[0,0].cpu().detach().numpy().squeeze()
                        angle_estm = output_angles[0,0].cpu().detach().numpy().squeeze()

                    else:
                        offset_estm = output_offsets[0,si].cpu().detach().numpy().squeeze()
                        angle_estm = output_angles[0,si].cpu().detach().numpy().squeeze()

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

                plt.imshow(image)
                plt.axis('off')
                plt.autoscale(False)

                l1.set_data([true_h1[0], true_h2[0]], [true_h1[1], true_h2[1]])
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

        plt.close()

        plt.figure()
        x = np.arange(0, len(all_offsets))
        all_offsets = np.array(all_offsets)
        all_offsets_estm = np.array(all_offsets_estm)
        # print(all_offsets)
        plt.plot(x, all_offsets, '-', c='#99C000')
        plt.plot(x, all_offsets_estm, '-', c='#0083CC')
        plt.ylim(-.4, .4)

        errors = np.abs(all_offsets-all_offsets_estm)
        err_mean = np.mean(errors).squeeze()
        err_stdd = np.std(errors).squeeze()
        plt.suptitle("error mean: %.4f -- stdd: %.4f" % (err_mean, err_stdd))

        plt.savefig(os.path.join(png_folder, "offsets_%03d.png" % idx), dpi=300)
        plt.savefig(os.path.join(svg_folder, "offsets_%03d.svg" % idx), dpi=300)
        plt.close()

