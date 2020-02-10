from resnet.resnet_plus_lstm import resnet18rnn
from datasets import kitti
from datasets import hlw
from utilities.tee import Tee
from torch import nn
from torchvision import transforms
from tensorboardX import SummaryWriter
import datetime
import os
import time
import platform
import shutil
import sklearn.metrics
import argparse
from utilities.losses import *
import random


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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--net', default='res18', type=str, metavar='NET', help='network type')
    parser.add_argument('--set', default='kitti', type=str, metavar='DS', help='dataset')
    parser.add_argument('--gpu', default='0', type=str, metavar='DS', help='dataset')
    parser.add_argument('--epochs', default=160, type=int, metavar='N', help='num epochs')
    parser.add_argument('--baselr', default=0.1 / 128, type=float, metavar='lr', help='base learning rate')
    parser.add_argument('--lr_reduction', default=1e-2, type=float, metavar='lr', help='base learning rate')
    parser.add_argument('--seqlength', default=32, type=int, metavar='N', help='sequence length')
    parser.add_argument('--seqlength_val', default=512, type=int, metavar='N', help='sequence length')
    parser.add_argument('--batch', default=4, type=int, metavar='B', help='batch size')
    parser.add_argument('--batch_val', default=1, type=int, metavar='B', help='batch size')
    parser.add_argument('--optimizer', default='sgd', type=str, metavar='optm', help='optimizer')
    parser.add_argument('--loss', default='huber', type=str, metavar='LF', help='loss function')
    parser.add_argument('--lossmax', default='l1', type=str, metavar='LF', help='loss function')
    parser.add_argument('--seed', default=1, type=int, metavar='S', help='random seed')
    parser.add_argument('--downscale', default=2, type=float, metavar='D', help='downscale factor')
    parser.add_argument('--cutout', default=512, type=int, help='use cutout', nargs='?', dest='cutout', const=512)
    parser.add_argument('--convlstm', dest='conv_lstm', action='store_true', help='use Conv LSTM layer')
    parser.add_argument('--workers', default=3, type=int, metavar='W', help='number of workers')
    parser.add_argument('--angle_loss_weight', default=1., type=float, metavar='S', help='random subsampling factor')
    parser.add_argument('--lstm_state_reduction', default=4., type=float, metavar='S', help='random subsampling factor')
    parser.add_argument('--lstm_depth', default=2, type=int, metavar='S', help='random subsampling factor')
    parser.add_argument('--lstm_mem', default=0, type=int, metavar='S', help='random subsampling factor')
    parser.add_argument('--load', default=None, type=str, metavar='DS', help='dataset')
    parser.add_argument('--eval', dest='eval', action='store_true', help='')
    parser.add_argument('--skip', dest='skip', action='store_true', help='')
    parser.add_argument('--bias', dest='bias', action='store_true', help='')
    parser.add_argument('--max_error_loss', dest='max_error_loss', action='store_true', help='')
    parser.add_argument('--max_error_loss_only', dest='max_error_loss_only', action='store_true', help='')
    parser.add_argument('--no_modelzoo_load', dest='nomzload', action='store_true', help='')
    parser.add_argument('--simple_skip', dest='simple_skip', action='store_true', help='')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    hostname = platform.node()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', 0)

    torch.backends.cudnn.deterministic = True

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
    np.random.seed(args.seed)
    random.seed(args.seed)

    if 'daidalos' in hostname:
        target_base = "/tnt/data/kluger/checkpoints/horizon_sequences"
        root_dir = "/tnt/data/kluger/datasets/kitti/horizons" if args.set == 'kitti' else "/tnt/data/scene_understanding/HLW"
        csv_base = "/tnt/home/kluger/tmp/kitti_split_%d" % 5
        pdf_file = "/tnt/home/kluger/tmp/kitti_split/data_pdfs.pkl"
    elif 'athene' in hostname:
        target_base = "/data/kluger/checkpoints/horizon_sequences"
        root_dir = "/phys/intern/kluger/tmp/kitti/horizons" if args.set == 'kitti' else "/phys/intern/kluger/tmp/HLW"
        csv_base = "/home/kluger/tmp/kitti_split_%d" % 5
        pdf_file = "/home/kluger/tmp/kitti_split/data_pdfs.pkl"
    elif 'hekate' in hostname:
        target_base = "/data/kluger/checkpoints/horizon_sequences"
        root_dir = "/phys/ssd/kitti/horizons" if args.set == 'kitti' else "/data/scene_understanding/HLW"
        csv_base = "/home/kluger/tmp/kitti_split_%d" % 5
        pdf_file = "/home/kluger/tmp/kitti_split/data_pdfs.pkl"
    elif 'persephone' in hostname or 'hades' in hostname:
        target_base = "/data/kluger/checkpoints/horizon_sequences"
        root_dir = "/phys/ssd/kluger/tmp/kitti/horizons" if args.set == 'kitti' else "/phys/ssd/kluger/tmp/HLW"
        csv_base = "/home/kluger/tmp/kitti_split_%d" % 5
        pdf_file = "/home/kluger/tmp/kitti_split/data_pdfs.pkl"
    else:
        target_base = "/data/kluger/checkpoints/horizon_sequences"
        root_dir = "/data/kluger/datasets/kitti/horizons" if args.set == 'kitti' else "/phys/ssd/kluger/tmp/HLW"
        csv_base = "/home/kluger/tmp/kitti_split_%d" % 5
        pdf_file = "/home/kluger/tmp/kitti_split/data_pdfs.pkl"

    if args.downscale > 1 and args.set == 'kitti':
        root_dir += "_s%.3f" % (1./args.downscale)

    pdf_file = None

    target_directory = target_base + "/%s/%s/d%d/%d/" % (args.set, args.net, args.downscale, args.seqlength)

    date_and_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")

    checkpoint_directory = target_directory + "b%d_" % args.batch + \
                           ("nocutout_" if (not args.cutout) else ("biasedcutout_" if (False) else "")) + date_and_time

    if not os.path.exists(checkpoint_directory):
        os.makedirs(checkpoint_directory)

    log_file = os.path.join(checkpoint_directory, "output.log")
    log = Tee(os.path.join(checkpoint_directory, log_file), "w", file_only=False)

    print("hostname: ", hostname)

    print(args)

    if args.net == 'res18':
        modelfun = resnet18rnn
    else:
        assert False

    model = modelfun(True, use_fc=False, use_convlstm=args.conv_lstm,
                     lstm_skip=args.skip,
                     lstm_bias=args.bias,
                     lstm_state_reduction=args.lstm_state_reduction, load=not(args.nomzload),
                     lstm_depth=args.lstm_depth,
                     lstm_simple_skip=args.simple_skip, lstm_mem=args.lstm_mem).to(device)

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
        scalemax = 0.1
    elif args.lossmax == 'huber':
        criterionmax = nn.SmoothL1Loss(size_average=False, reduce=False)
        scalemax = 0.1
    elif args.lossmax == 'l1':
        criterionmax = nn.L1Loss(size_average=False, reduce=False)
        scalemax = 0.1
    elif args.lossmax == 'sqrt':
        criterionmax = SqrtL1Loss(size_average=False, reduce=False)
        scalemax = 0.1#0.02
    else:
        assert False

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-4, momentum=0.9)
    else:
        assert False

    horizon_error_function = horizon_error(WIDTH, HEIGHT)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, int(args.epochs), eta_min=learning_rate*args.lr_reduction)

    max_err_scheduler = CosineAnnealingCustom(0, 1., args.epochs)

    tfs = transforms.Compose([
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=pixel_mean, std=[1., 1., 1.]),
            ])
    if args.cutout > 0:
        tfs.transforms.append(DS.Cutout(length=args.cutout, bias=False))

    tfs_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=pixel_mean, std=[1., 1., 1.]),
            ])

    if args.set == 'kitti':

        train_dataset = DS.KittiRawDatasetPP(root_dir=root_dir, pdf_file=pdf_file, random_subsampling=1.,
                                             csv_file=csv_base + "/train.csv", seq_length=args.seqlength,
                                             im_height=HEIGHT, im_width=WIDTH, fill_up=True,
                                             scale=1./args.downscale, transform=tfs, get_split_data=False,
                                             overlap=0, zero_start=False)
        val_dataset = DS.KittiRawDatasetPP(root_dir=root_dir, pdf_file=pdf_file, augmentation=False,
                                           csv_file=csv_base + "/val.csv", seq_length=args.seqlength_val,
                                           im_height=HEIGHT, im_width=WIDTH, fill_up=False,
                                           scale=1./args.downscale, transform=tfs_val, get_split_data=False,
                                           zero_start=False)
    elif args.set == 'hlw':

        train_dataset = DS.HLWDataset(root_dir=root_dir, transform=tfs, augmentation=True, set='train', scale=1./args.downscale)
        val_dataset = DS.HLWDataset(root_dir=root_dir, augmentation=False, transform=tfs_val, set='val', scale=1./args.downscale)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch,
                                               shuffle=True, num_workers=workers)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=args.batch_val,
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

    print(checkpoint_directory)

    # Train the model
    total_step = len(train_loader)
    curr_lr = learning_rate
    curr_batch_size = args.batch
    best_val_loss = 10000.

    calc_hlr = calc_horizon_leftright(width=WIDTH, height=HEIGHT)

    best_auc = {'epoch': 0, 'max_err': np.inf, 'auc':0}
    best_err = {'epoch': 0, 'max_err': np.inf, 'auc':0}

    tensorboard_directory = checkpoint_directory + "/tensorboard/"
    if not os.path.exists(tensorboard_directory):
        os.makedirs(tensorboard_directory)
    tensorboard_writer = SummaryWriter(tensorboard_directory)

    for epoch in range(args.epochs):


        if not args.eval:
            scheduler.step()
            adjust_learning_rate(optimizer, scheduler.get_lr()[0])

            losses = []
            offset_losses = []
            angle_losses = []
            offset_dif_losses = []
            angle_dif_losses = []
            temp_offset_losses = []
            temp_angle_losses = []
            max_err_losses = []

            tt0 = time.time()

            model.train()
            for i, sample in enumerate(train_loader):

                images = sample['images'].to(device, non_blocking=True)
                offsets = sample['offsets'].to(device, non_blocking=True)
                angles = sample['angles'].to(device, non_blocking=True)

                output_offsets, output_angles = model(images)

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

                if args.max_error_loss:
                    if args.max_error_loss_only:
                        loss = max_err_loss
                    else:
                        loss = max_err_scheduler.get(epoch) * max_err_loss * scalemax + (1-max_err_scheduler.get(epoch)) * loss

                tt3 = time.time()

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                tt4 = time.time()
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

                    num_iteration = int((epoch*total_step + i) * images_per_batch / 128.)

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
                    tensorboard_writer.add_scalar('learning_rate', scheduler.get_lr()[0], num_iteration)

                tt0 = time.time()

        # test on validation set:
        model.eval()
        with torch.no_grad():
            losses = []
            offset_losses = []
            angle_losses = []
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

                image_count += images.shape[0]*images.shape[1]

                if args.ema > 0:
                    output_offsets_dif, output_angles_dif, output_offsets_ema, output_angles_ema = \
                        model(images)
                else:
                    output_offsets, output_angles = \
                        model(images)

                offset_loss = criterion(output_offsets, offsets)
                angle_loss = criterion(output_angles, angles)

                loss = offset_loss + angle_loss * args.angle_loss_weight

                hl_true, hr_true = calc_hlr(offsets, angles)
                hl_estm, hr_estm = calc_hlr(output_offsets, output_angles)
                hl_err = criterionmax(hl_estm, hl_true)
                hr_err = criterionmax(hr_estm, hr_true)
                h_errs = torch.max(hl_err, hr_err)
                max_err_loss = torch.mean(h_errs)
                max_err_losses += [max_err_loss]

                if args.max_error_loss:
                    if args.max_error_loss_only:
                        loss = max_err_loss * scalemax
                    else:
                        loss = max_err_scheduler.get(epoch) * scalemax * max_err_loss + (1 - max_err_scheduler.get(epoch)) * loss

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

            num_iteration = int((epoch * total_step + idx) * images_per_batch / 128.)

            max_err_losses_tensor = torch.stack(max_err_losses, dim=0).view(-1)
            average_max_err_loss = max_err_losses_tensor.mean().item()
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

            if args.eval:
                exit(0)

            tensorboard_writer.add_scalar('val/loss_avg', average_loss, num_iteration)
            tensorboard_writer.add_scalar('val/offset_loss_avg', average_offset_loss, num_iteration)
            tensorboard_writer.add_scalar('val/angle_loss_avg', average_angle_loss, num_iteration)
            tensorboard_writer.add_scalar('val/auc', auc, num_iteration)

        is_best = (average_loss < best_val_loss)
        best_val_loss = average_loss if is_best else best_val_loss

        save_checkpoint({
                'epoch': epoch,
                'args': args,
                'state_dict': model.module.state_dict(),
                'val_loss': average_loss,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint_directory, epoch, average_loss)


    tensorboard_writer.close()
    log.__del__()

