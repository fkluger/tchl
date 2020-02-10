from resnet import resnet_plus_lstm
from resnet_ar import resnet_ar_models
from datasets import kitti
from datasets import hlw
from utilities.tee import Tee
import torch
from torch import nn
from torchvision import transforms
from tensorboardX import SummaryWriter
import datetime
import os
import numpy as np
import math
import time
import platform
import shutil
import sklearn.metrics
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F
import argparse
from utilities.losses import *

#torch.backends.cudnn.benchmark = True

class CosineAnnealingCustom:

    def __init__(self, begin, end, T_max):
        self.T_max = T_max
        self.begin = begin
        self.end = end
        self.inv = begin < end

    def get(self, epoch):
        if not self.inv:
            return self.end + (self.begin - self.end) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        else:
            return self.begin + (self.end - self.begin) * (1 - math.cos(math.pi * epoch / self.T_max)) / 2


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def horizon_error(width, height):

    calc_hlr = calc_horizon_leftright(width, height)

    def f(estm_ang, estm_off, true_ang, true_off):
        errors = []

        for b in range(estm_ang.shape[0]):
            for s in range(estm_ang.shape[1]):

                offset = true_off[b,s].squeeze()
                offset_estm = estm_off[b,s].squeeze()
                angle = true_ang[b,s].squeeze()
                angle_estm = estm_ang[b,s].squeeze()

                ylt, yrt = calc_hlr(offset, angle)
                yle, yre = calc_hlr(offset_estm, angle_estm)

                err1 = np.abs((ylt-yle).cpu().detach().numpy())
                err2 = np.abs((yrt-yre).cpu().detach().numpy())

                err = np.maximum(err1, err2)
                errors += [err]

        return errors

    return f


def calc_horizon_leftright(width, height):
    wh = 0.5 * width*1./height

    def f(offset, angle):
        term2 = wh * torch.tan(torch.clamp(angle, -math.pi/3., math.pi/3.))
        return offset + 0.5 + term2, offset + 0.5 - term2

    return f


class TemporalMSELoss(_Loss):
    def __init__(self, size_average=True, reduce=True, reduction='elementwise_mean', from_start=False):
        super(TemporalMSELoss, self).__init__(size_average, reduce, reduction)
        self.from_start = from_start

    def forward(self, input, target):

        S = input.shape[1]

        input_diffs = []
        target_diffs = []

        if self.from_start:
            for s in range(1,S):
                input_diffs += [input[:,s,:]-input[:,0,:]]
                target_diffs += [target[:,s,:]-target[:,0,:]]
        else:
            for s in range(1,S):
                input_diffs += [input[:,s,:]-input[:,s-1,:]]
                target_diffs += [target[:,s,:]-target[:,s-1,:]]

        target_diffs = torch.stack(target_diffs, dim=1)
        input_diffs = torch.stack(input_diffs, dim=1)

        return F.mse_loss(input_diffs, target_diffs, reduction=self.reduction)


class MaxErrorLoss(_Loss):
    def __init__(self, size_average=True, reduce=True, reduction='elementwise_mean', from_start=False):
        super(MaxErrorLoss, self).__init__(size_average, reduce, reduction)
        self.from_start = from_start

    def forward(self, input, target):

        S = input.shape[1]

        input_diffs = []
        target_diffs = []

        if self.from_start:
            for s in range(1,S):
                input_diffs += [input[:,s,:]-input[:,0,:]]
                target_diffs += [target[:,s,:]-target[:,0,:]]
        else:
            for s in range(1,S):
                input_diffs += [input[:,s,:]-input[:,s-1,:]]
                target_diffs += [target[:,s,:]-target[:,s-1,:]]

        target_diffs = torch.stack(target_diffs, dim=1)
        input_diffs = torch.stack(input_diffs, dim=1)

        return F.mse_loss(input_diffs, target_diffs, reduction=self.reduction)


class CalcConfidenceTarget(torch.nn.Module):
    def __init__(self, max_error, device):
        super(CalcConfidenceTarget, self).__init__()
        self.max_error = max_error
        self.device = device

    def forward(self, input, target):

        input = torch.squeeze(input)
        target = torch.squeeze(target)

        # print("input shape: ", input.shape)
        # exit(0)

        ones = torch.ones(input.shape, dtype=torch.long).to(self.device)
        zero = torch.zeros(input.shape, dtype=torch.long).to(self.device)

        diffs = (input-target)
        diffs = diffs*diffs

        target = torch.where(diffs < self.max_error, zero, ones)

        return target.detach()


class FeatMapBuffer:

    def __init__(self, length, shape1=None, shape2=None):
        self.length = length
        self.shape1 = shape1
        self.shape2 = shape2
        if shape1 is not None and shape2 is not None:
            self.queue1 = np.zeros((length,) + shape1, dtype=np.float32)
            self.queue2 = np.zeros((length,) + shape2, dtype=np.float32)
        else:
            self.queue1 = None
            self.queue2 = None
        self.idx = 0

    def enqueue(self, feat1, feat2):
        len1 = feat1.shape[0] * feat1.shape[1]
        len2 = feat2.shape[0] * feat2.shape[1]
        assert len1 == len2
        length = len1
        feat1 = feat1.reshape((length, feat1.shape[2], feat1.shape[3], feat1.shape[4]))
        feat2 = feat2.reshape((length, feat2.shape[2], feat2.shape[3], feat2.shape[4]))

        if self.shape1 is None and self.shape2 is None:
            self.shape1 = tuple(feat1.shape[1:])
            self.shape2 = tuple(feat2.shape[1:])

            self.queue1 = np.zeros((self.length,) + self.shape1)
            self.queue2 = np.zeros((self.length,) + self.shape2)

        for si in range(length):
            self.queue1[self.idx,:,:,:] = feat1[si,:,:,:]
            self.queue2[self.idx,:,:,:] = feat2[si,:,:,:]
            self.idx = (self.idx+1) % self.length

    def get_random_features(self, num, batches):
        feat1 = np.zeros((batches, num,) + self.shape1, dtype=np.float32)
        feat2 = np.zeros((batches, num,) + self.shape2, dtype=np.float32)
        for bi in range(batches):
            for si in range(num):
                idx = np.random.randint(0, self.length)
                feat1[bi,si,:,:,:] = self.queue1[idx,:,:,:]
                feat2[bi,si,:,:,:] = self.queue2[idx,:,:,:]

        return feat1, feat2

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--net', default='resnet18ar', type=str, metavar='NET', help='network type')
    parser.add_argument('--set', default='kitti', type=str, metavar='DS', help='dataset')
    parser.add_argument('--gpu', default='0', type=str, metavar='DS', help='dataset')
    parser.add_argument('--finetune', dest='finetune', action='store_true', help='finetune the CNN')
    parser.add_argument('--epochs', default=128, type=int, metavar='N', help='num epochs')
    parser.add_argument('--baselr', default=0.1 / 128, type=float, metavar='lr', help='base learning rate')
    parser.add_argument('--lr_reduction', default=1e-2, type=float, metavar='lr', help='base learning rate')
    parser.add_argument('--seqlength', default=1, type=int, metavar='N', help='sequence length')
    parser.add_argument('--seqlength_val', default=512, type=int, metavar='N', help='sequence length')
    parser.add_argument('--batch', default=8 * 16, type=int, metavar='B', help='batch size')
    # parser.add_argument('--batch_val', default=8 * 16, type=int, metavar='B', help='batch size')
    parser.add_argument('--optimizer', default='sgd', type=str, metavar='optm', help='optimizer')
    parser.add_argument('--loss', default='huber', type=str, metavar='LF', help='loss function')
    parser.add_argument('--lossmax', default='l1', type=str, metavar='LF', help='loss function')
    parser.add_argument('--seed', default=1, type=int, metavar='S', help='random seed')
    parser.add_argument('--downscale', default=2, type=float, metavar='D', help='downscale factor')
    parser.add_argument('--cutout', dest='cutout', action='store_true', help='use cutout')
    parser.add_argument('--temploss', dest='temporal_loss', action='store_true', help='use temporal loss')
    parser.add_argument('--temploss2', dest='temporal_loss_2', action='store_true', help='use temporal loss')
    parser.add_argument('--templossonly', dest='temporal_loss_only', action='store_true', help='use temporal loss')
    parser.add_argument('--workers', default=3, type=int, metavar='W', help='number of workers')
    parser.add_argument('--random_subsampling', default=1., type=float, metavar='S', help='random subsampling factor')
    parser.add_argument('--angle_loss_weight', default=1., type=float, metavar='S', help='random subsampling factor')
    parser.add_argument('--load', default=None, type=str, metavar='DS', help='dataset')
    parser.add_argument('--eval', dest='eval', action='store_true', help='')
    parser.add_argument('--overlap', default=0, type=int, metavar='S', help='random subsampling factor')
    parser.add_argument('--feat_buffer_length', default=128, type=int, metavar='S', help='random subsampling factor')
    parser.add_argument('--feat_buffer_samples', default=4, type=int, metavar='S', help='random subsampling factor')
    parser.add_argument('--max_error_loss', dest='max_error_loss', action='store_true', help='')
    parser.add_argument('--max_error_loss_only', dest='max_error_loss_only', action='store_true', help='')
    parser.add_argument('--no_fill_up', dest='no_fill_up', action='store_true', help='')
    parser.add_argument('--fp16', dest='fp16', action='store_true', help='')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    hostname = platform.node()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', 0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    if args.set == 'kitti':
        DS = kitti
        pixel_mean = [0.362365, 0.377767, 0.366744]
    elif args.set == 'hlw':
        DS = hlw
        pixel_mean = [0.469719773, 0.462005855, 0.454649294]
    else:
        assert False

    WIDTH = int(DS.WIDTH // args.downscale)
    HEIGHT = int(DS.HEIGHT // args.downscale)

    learning_rate = args.baselr * args.batch * args.seqlength

    images_per_batch = args.batch * args.seqlength

    workers = args.workers

    torch.manual_seed(args.seed)
    if 'daidalos' in hostname:
        target_base = "/tnt/data/kluger/checkpoints/horizon_sequences"
        root_dir = "/tnt/data/kluger/datasets/kitti/horizons" if args.set == 'kitti' else "/tnt/data/scene_understanding/HLW"
        csv_base = "/tnt/home/kluger/tmp/kitti_split_3"
        pdf_file = "/tnt/home/kluger/tmp/kitti_split/data_pdfs.pkl"
    elif 'athene' in hostname:
        target_base = "/data/kluger/checkpoints/horizon_sequences"
        root_dir = "/phys/intern/kluger/tmp/kitti/horizons" if args.set == 'kitti' else "/data/scene_understanding/HLW"
        csv_base = "/home/kluger/tmp/kitti_split_3"
        pdf_file = "/home/kluger/tmp/kitti_split/data_pdfs.pkl"
    elif 'hekate' in hostname:
        target_base = "/data/kluger/checkpoints/horizon_sequences"
        root_dir = "/phys/ssd/kitti/horizons" if args.set == 'kitti' else "/data/scene_understanding/HLW"
        csv_base = "/home/kluger/tmp/kitti_split_3"
        pdf_file = "/home/kluger/tmp/kitti_split/data_pdfs.pkl"
    else:
        target_base = "/data/kluger/checkpoints/horizon_sequences"
        root_dir = "/phys/ssd/kluger/tmp/kitti/horizons" if args.set == 'kitti' else "/phys/ssd/kluger/tmp/HLW"
        csv_base = "/home/kluger/tmp/kitti_split_3"
        pdf_file = "/home/kluger/tmp/kitti_split/data_pdfs.pkl"

    if args.downscale > 1:
        root_dir += "_s%.3f" % (1./args.downscale)

    pdf_file = None

    if args.net == 'resnet18ar':
        modelfun = resnet_ar_models.resnet_ar
    else:
        assert False

    model, blocks = modelfun(order='BDCHW')
    model = model.to(device)

    fov_increase = model.fov_increase
    overlap = 2*fov_increase if args.overlap == 0 else args.overlap

    if args.fp16:
        model.half()  # convert to half precision
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    target_directory = target_base + "/%s/%s_3d/d%d/%d_%d/" % (args.set, args.net, args.downscale, overlap, args.seqlength)

    date_and_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")

    checkpoint_directory = target_directory + "b%d_" % args.batch + ("nocutout_" if (not args.cutout) else ("biasedcutout_" if (False) else "")) + date_and_time
    tensorboard_directory = checkpoint_directory + "/tensorboard/"
    if not os.path.exists(tensorboard_directory):
        os.makedirs(tensorboard_directory)

    log_file = os.path.join(checkpoint_directory, "output.log")
    log = Tee(os.path.join(checkpoint_directory, log_file), "w", file_only=False)

    print("hostname: ", hostname)

    # print(args)
    for arg in vars(args):
        print(arg, getattr(args, arg))

    print("fov increase: ", model.fov_increase)

    for b in blocks: print(b.block_name)

    if args.load is not None:
        load_from_path = args.load
        print("load weights from ", load_from_path)
        checkpoint = torch.load(load_from_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'], strict=True)

    model = nn.DataParallel(model)

    if args.loss == 'mse':
        criterion = nn.MSELoss()
    elif args.loss == 'huber':
        criterion = nn.SmoothL1Loss()
    elif args.loss == 'l1':
        criterion = nn.L1Loss()
    else:
        assert False

    if args.lossmax == 'mse':
        criterionmax = nn.MSELoss(size_average=False, reduce=False)
    elif args.lossmax == 'huber':
        criterionmax = nn.SmoothL1Loss(size_average=False, reduce=False)
    elif args.lossmax == 'l1':
        criterionmax = nn.L1Loss(size_average=False, reduce=False)
    elif args.lossmax == 'sqrt':
        criterionmax = SqrtL1Loss(size_average=False, reduce=False)
    else:
        assert False

    temp_criterion = TemporalMSELoss(from_start=args.temporal_loss_2)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-4, momentum=0.9)
    else:
        assert False

    horizon_error_function = horizon_error(WIDTH, HEIGHT)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, int(args.epochs), eta_min=learning_rate*args.lr_reduction)

    # max_err_scheduler = CosineAnnealingCustom(0, 0.1, args.epochs)
    max_err_scheduler = CosineAnnealingCustom(0, 1., args.epochs)

    tfs = transforms.Compose([
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=pixel_mean, std=[1., 1., 1.]),
            ])
    if args.cutout:
        tfs.transforms.append(DS.Cutout(length=512 if args.set == 'kitti' else 256, bias=False))

    tfs_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=pixel_mean, std=[1., 1., 1.]),
            ])

    if args.set == 'kitti':

        train_dataset = DS.KittiRawDatasetPP(root_dir=root_dir, pdf_file=pdf_file, random_subsampling=args.random_subsampling,
                                             csv_file=csv_base + "/train.csv", seq_length=args.seqlength,
                                             im_height=HEIGHT, im_width=WIDTH, fill_up=(not args.no_fill_up),
                                             scale=1./args.downscale, transform=tfs, pre_padding=2*fov_increase,
                                             overlap=overlap, )
        val_dataset = DS.KittiRawDatasetPP(root_dir=root_dir, pdf_file=pdf_file, augmentation=False,
                                           csv_file=csv_base + "/val.csv", seq_length=args.seqlength_val,
                                           im_height=HEIGHT, im_width=WIDTH, fill_up=False, overlap=overlap,
                                           scale=1./args.downscale, transform=tfs_val, pre_padding=2*fov_increase,)
    elif args.set == 'hlw':

        train_dataset = DS.HLWDataset(root_dir=root_dir, transform=tfs, augmentation=True, set='train')
        val_dataset = DS.HLWDataset(root_dir=root_dir, augmentation=False, transform=tfs_val, set='val')

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch,
                                               shuffle=True, num_workers=workers)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=1,
                                              shuffle=False, num_workers=workers)

    # For updating learning rate
    def update_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def update_batchsize(loader, batch_size):
        loader = torch.utils.data.DataLoader(dataset=loader.dataset, batch_size=batch_size, shuffle=True, num_workers=loader.num_workers)
        return loader

    def save_checkpoint(state, is_best, folder, epoch, loss):
        filename = folder + "/" + "%03d_%.6f.ckpt" % (epoch, loss)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, folder + '/model_best.ckpt')

    tensorboard_writer = SummaryWriter(tensorboard_directory)

    print(checkpoint_directory)

    if args.temporal_loss_only:
        assert False

    # Train the model
    total_step = len(train_loader)
    curr_lr = learning_rate
    curr_batch_size = args.batch
    best_val_loss = 10000.

    calc_hlr = calc_horizon_leftright(width=WIDTH, height=HEIGHT)

    best_auc = {'epoch': 0, 'max_err': np.inf, 'auc':0}
    best_err = {'epoch': 0, 'max_err': np.inf, 'auc':0}

    feature_map_buffer = FeatMapBuffer(args.feat_buffer_length)

    i = 0

    for epoch in range(args.epochs):

        # if config.use_dropblock:
        #     model.set_dropblock_prob(1-1.*epoch/config.num_epochs*config.dropblock_drop_prob)

        if not args.eval:
            scheduler.step()
            adjust_learning_rate(optimizer, scheduler.get_lr()[0])

            losses = []
            offset_losses = []
            angle_losses = []
            temp_offset_losses = []
            temp_angle_losses = []
            max_err_losses = []

            tt0 = time.time()

            model.train()
            for i, sample in enumerate(train_loader):

                # if i > 400: break
                images = sample['images'].to(device, non_blocking=True)
                offsets = sample['offsets'].to(device, non_blocking=True)
                angles = sample['angles'].to(device, non_blocking=True)

                if feature_map_buffer.shape1 is not None and args.feat_buffer_samples > 0:
                    yo, y_normed = feature_map_buffer.get_random_features(args.feat_buffer_samples, args.batch)
                    # print(yo.dtype, y_normed.dtype)
                    yo = torch.from_numpy(yo).to(device)
                    y_normed = torch.from_numpy(y_normed).to(device)
                    # print(yo.dtype, y_normed.dtype)

                else:
                    yo = None
                    y_normed = None

                # Forward pass
                output_offsets, output_angles, yo, y_normed = model(images, yo, y_normed)
                output_offsets = output_offsets[:,fov_increase:args.seqlength+fov_increase]
                output_angles = output_angles[:,fov_increase:args.seqlength+fov_increase]

                if args.feat_buffer_samples > 0:
                    feature_map_buffer.enqueue(yo.detach().cpu().numpy(), y_normed.detach().cpu().numpy())

                offset_loss = criterion(output_offsets, offsets)
                angle_loss = criterion(output_angles, angles)

                loss = 0

                # if args.max_error_loss:
                hl_true, hr_true = calc_hlr(offsets, angles)
                hl_estm, hr_estm = calc_hlr(output_offsets, output_angles)
                hl_err = criterionmax(hl_estm, hl_true)
                hr_err = criterionmax(hr_estm, hr_true)
                h_errs = torch.clamp(torch.max(hl_err, hr_err), 0, 1.)
                max_err_loss = torch.mean(h_errs)
                max_err_losses += [max_err_loss]

                loss += offset_loss + angle_loss * args.angle_loss_weight

                if args.temporal_loss:
                    temp_offset_loss = temp_criterion(output_offsets, offsets)
                    temp_angle_loss = temp_criterion(output_angles, angles)
                    loss += temp_offset_loss + temp_angle_loss * args.angle_loss_weight
                    temp_offset_losses.append(temp_offset_loss)
                    temp_angle_losses.append(temp_angle_loss)

                if args.max_error_loss:
                    if args.max_error_loss_only:
                        loss = max_err_loss
                    else:
                        # loss = max_err_scheduler.get(epoch) * max_err_loss + (1-max_err_scheduler.get(epoch)) * loss
                        loss = max_err_scheduler.get(epoch) * max_err_loss * 0.1 + (1-max_err_scheduler.get(epoch)) * loss

                tt3 = time.time()


                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                tt4 = time.time()
                # if config.gradient_clip > 0:
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                tt5 = time.time()

                # print('data loading : %f' % (tt01-tt0) )
                # print('data to dev. : %f' % (tt1-tt01) )
                # print('forward pass : %f' % (tt2-tt1) )
                # print('loss calculat: %f' % (tt3-tt2) )
                # print('backward pass: %f' % (tt4-tt3) )
                # print('optimization : %f' % (tt5-tt4) )
                # print("...")

                # losses.append(loss.item())
                losses.append(loss)
                offset_losses.append(offset_loss)
                angle_losses.append(angle_loss)

                if (i+1) % 100 == 0:
                    # average_loss = np.mean(losses)
                    losses_tensor = torch.stack(losses, dim=0).view(-1)
                    average_loss = losses_tensor.mean().item()

                    offset_losses_tensor = torch.stack(offset_losses, dim=0).view(-1)
                    average_offset_loss = offset_losses_tensor.mean().item()
                    angle_losses_tensor = torch.stack(angle_losses, dim=0).view(-1)
                    average_angle_loss = angle_losses_tensor.mean().item()

                    # average_offset_loss = np.mean(offset_losses)
                    # average_angle_loss = np.mean(angle_losses)

                    num_iteration = int((epoch*total_step + i) * images_per_batch / 128.)

                    if args.temporal_loss:
                        # temp_average_offset_loss = np.mean(temp_offset_losses)
                        # temp_average_angle_loss = np.mean(temp_angle_losses)
                        temp_offset_losses_tensor = torch.stack(temp_offset_losses, dim=0).view(-1)
                        temp_average_offset_loss = temp_offset_losses_tensor.mean().item()
                        temp_angle_losses_tensor = torch.stack(temp_angle_losses, dim=0).view(-1)
                        temp_average_angle_loss = temp_angle_losses_tensor.mean().item()

                    # if args.max_error_loss:
                    max_err_losses_tensor = torch.stack(max_err_losses, dim=0).view(-1)
                    average_max_err_loss = max_err_losses_tensor.mean().item()
                    tensorboard_writer.add_scalar('train/max_err_loss', max_err_loss.item(), num_iteration)
                    tensorboard_writer.add_scalar('train/max_err_loss_avg', average_max_err_loss, num_iteration)


                    print ("Epoch [{}/{}], Step [{}/{}] Losses: {:.6f} {:.6f} {:.6f}, Avg.: {:.6f} {:.6f} {:.6f}"
                           .format(epoch+1, args.epochs, i+1, total_step, offset_loss.item(), angle_loss.item(), loss.item(),
                                   average_offset_loss, average_angle_loss, average_loss), end="\r")

                    tensorboard_writer.add_scalar('train/loss', loss.item(), num_iteration)
                    tensorboard_writer.add_scalar('train/offset_loss', offset_loss.item(), num_iteration)
                    tensorboard_writer.add_scalar('train/angle_loss', angle_loss.item(), num_iteration)
                    tensorboard_writer.add_scalar('train/loss_avg', average_loss, num_iteration)
                    tensorboard_writer.add_scalar('train/offset_loss_avg', average_offset_loss, num_iteration)
                    tensorboard_writer.add_scalar('train/angle_loss_avg', average_angle_loss, num_iteration)
                    if args.temporal_loss:
                        tensorboard_writer.add_scalar('train/temp_offset_loss', temp_offset_loss.item(), num_iteration)
                        tensorboard_writer.add_scalar('train/temp_angle_loss', temp_angle_loss.item(), num_iteration)
                        tensorboard_writer.add_scalar('train/temp_offset_loss_avg', temp_average_offset_loss, num_iteration)
                        tensorboard_writer.add_scalar('train/temp_angle_loss_avg', temp_average_angle_loss, num_iteration)

                    tensorboard_writer.add_scalar('learning_rate', scheduler.get_lr()[0], num_iteration)

                tt0 = time.time()


        # test on validation set:
        model.eval()
        with torch.no_grad():
            losses = []
            offset_losses = []
            angle_losses = []
            offset_ema_losses = []
            angle_ema_losses = []
            offset_dif_losses = []
            angle_dif_losses = []
            temp_offset_losses = []
            temp_angle_losses = []
            max_err_losses = []

            all_horizon_errors = []

            image_count = 0

            for idx, sample in enumerate(val_loader):
                images = sample['images'].to(device)
                offsets = sample['offsets'].to(device)
                angles = sample['angles'].to(device)
                    
                # print(images.shape)
                image_count += images.shape[0]*images.shape[1]

                output_offsets, output_angles, _, _ = model(images)
                output_offsets = output_offsets[:,fov_increase:offsets.shape[1]+fov_increase]
                output_angles = output_angles[:,fov_increase:angles.shape[1]+fov_increase]

                offset_loss = criterion(output_offsets, offsets)
                angle_loss = criterion(output_angles, angles)

                loss = offset_loss + angle_loss * args.angle_loss_weight
                    
                if args.temporal_loss:
                    temp_offset_loss = temp_criterion(output_offsets, offsets)
                    temp_angle_loss = temp_criterion(output_angles, angles)
                    loss += temp_offset_loss + temp_angle_loss * args.angle_loss_weight
                    temp_offset_losses.append(temp_offset_loss.item())
                    temp_angle_losses.append(temp_angle_loss.item())


                # if args.max_error_loss:
                hl_true, hr_true = calc_hlr(offsets, angles)
                hl_estm, hr_estm = calc_hlr(output_offsets, output_angles)
                hl_err = criterionmax(hl_estm, hl_true)
                hr_err = criterionmax(hr_estm, hr_true)
                h_errs = torch.max(hl_err, hr_err)
                max_err_loss = torch.mean(h_errs)
                # loss += max_err_loss
                max_err_losses += [max_err_loss]

                if args.max_error_loss:
                    if args.max_error_loss_only:
                        loss = max_err_scheduler.get(epoch) * max_err_loss + (1 - max_err_scheduler.get(epoch)) * loss
                    else:
                        loss = max_err_scheduler.get(epoch) * 0.1 * max_err_loss + (1 - max_err_scheduler.get(epoch)) * loss

                all_horizon_errors += horizon_error_function(output_angles,
                                                             output_offsets,
                                                             angles,
                                                             offsets)
                losses.append(loss.item())
                offset_losses.append(offset_loss.item())
                angle_losses.append(angle_loss.item())

            average_loss = np.mean(losses)
            average_offset_loss = np.mean(offset_losses)
            average_angle_loss = np.mean(angle_losses)

            if args.temporal_loss:
                temp_average_offset_loss = np.mean(temp_offset_losses)
                temp_average_angle_loss = np.mean(temp_angle_losses)

            # num_iteration = epoch * total_step

            num_iteration = int((epoch * total_step + i) * images_per_batch / 128.)

            # if args.max_error_loss:
            max_err_losses_tensor = torch.stack(max_err_losses, dim=0).view(-1)
            average_max_err_loss = max_err_losses_tensor.mean().item()
            # tensorboard_writer.add_scalar('val/max_err_loss', max_err_loss.item(), num_iteration)
            tensorboard_writer.add_scalar('val/max_err_loss_avg', average_max_err_loss, num_iteration)

            error_arr = np.array(all_horizon_errors)
            error_arr_idx = np.argsort(error_arr)
            error_arr = np.sort(error_arr)
            num_values = len(all_horizon_errors)

            plot_points = np.zeros((num_values, 2))

            err_cutoff = 0.25

            midfraction = 1.

            try:
                for i in range(num_values):
                    fraction = (i + 1) * 1.0 / num_values
                    value = error_arr[i]
                    plot_points[i, 1] = fraction
                    plot_points[i, 0] = value
                    if i > 0:
                        lastvalue = error_arr[i - 1]
                        if lastvalue < err_cutoff and value > err_cutoff:
                            midfraction = (lastvalue * plot_points[i - 1, 1] + value * fraction) / (value + lastvalue)

                if plot_points[-1, 0] < err_cutoff:
                    plot_points = np.vstack([plot_points, np.array([err_cutoff, 1])])
                else:
                    # print("midfraction: ", midfraction)
                    plot_points = np.vstack([plot_points, np.array([err_cutoff, midfraction])])

                sorting = np.argsort(plot_points[:, 0])
                plot_points = plot_points[sorting, :]

                auc = sklearn.metrics.auc(plot_points[plot_points[:, 0] <= err_cutoff, 0],
                                          plot_points[plot_points[:, 0] <= err_cutoff, 1])
                auc = auc / err_cutoff
            except:
                auc = 0

            print("\nValidation [{}/{}],  Avg.: {:.4f} {:.4f} {:.4f} {:.4f}"
                  .format(epoch + 1, args.epochs, average_offset_loss, average_angle_loss, average_loss, auc))

            if best_err['max_err'] > average_max_err_loss:
                best_err['epoch'] = epoch
                best_err['max_err'] = average_max_err_loss
                best_err['auc'] = auc

            if best_auc['auc'] < auc:
                best_auc['epoch'] = epoch
                best_auc['max_err'] = average_max_err_loss
                best_auc['auc'] = auc

            print("Best Err: %.4f -- AUC: %.4f -- epoch %d" % (best_err['max_err'], best_err['auc'], best_err['epoch']))
            print("Best AUC: %.4f -- Err: %.4f -- epoch %d" % (best_auc['auc'], best_auc['max_err'], best_auc['epoch']))

            #print("%d images" % image_count)
            #print(error_arr.shape)
            #print(np.mean(error_arr))
            #print("errors: \n", error_arr)
            #print(error_arr_idx)

            if args.eval:
                exit(0)

            tensorboard_writer.add_scalar('val/loss_avg', average_loss, num_iteration)
            tensorboard_writer.add_scalar('val/offset_loss_avg', average_offset_loss, num_iteration)
            tensorboard_writer.add_scalar('val/angle_loss_avg', average_angle_loss, num_iteration)

            if args.temporal_loss:
                tensorboard_writer.add_scalar('val/temp_offset_loss_avg', temp_average_offset_loss, num_iteration)
                tensorboard_writer.add_scalar('val/temp_angle_loss_avg', temp_average_angle_loss, num_iteration)
            tensorboard_writer.add_scalar('val/auc', auc, num_iteration)

        # torch.save(model.state_dict(), checkpoint_directory + "/" + "%03d_%.6f.ckpt" % (epoch, average_loss))

        is_best = (average_loss < best_val_loss)
        best_val_loss = average_loss if is_best else best_val_loss

        save_checkpoint({
                'epoch': epoch,
                'args': args,
                'state_dict': model.module.state_dict(),
                'val_loss': average_loss,
                'optimizer' : optimizer.state_dict(),
                # 'all_gradients' : np.concatenate([x.grad.cpu().detach().numpy().flatten() for x in model.parameters() if x is not None])
            }, is_best, checkpoint_directory, epoch, average_loss)

        # if config.batch_size_updates is not None:
        #     # Increase batch size
        #     if (epoch+1) in config.batch_size_updates:
        #         curr_batch_size *= config.batch_size_multi
        #         train_loader = update_batchsize(train_loader, curr_batch_size)
        #         print("new batch size: ", train_loader.batch_size)
        # if config.learning_rate_updates is not None:
        #     # Decay learning rate
        #     if (epoch+1) in config.learning_rate_updates == 0:
        #         curr_lr *= config.learning_rate_multi
        #         update_lr(optimizer, curr_lr)
        #         print("new learning rate: ", train_loader.batch_size)

    tensorboard_writer.close()
    log.__del__()

