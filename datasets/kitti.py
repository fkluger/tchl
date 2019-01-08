from torch.utils.data import Dataset
import csv
import pykitti
import numpy as np
import random
import time
import pickle
from scipy import ndimage
import glob
import torch
import cv2
from PIL import Image

WIDTH = 1250
HEIGHT = 380


class KittiRawDataset(Dataset):

    def __init__(self, csv_file, root_dir, seq_length, augmentation=True, pdf_file=None):

        self.sequences = []

        with open(csv_file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                date = row[0]
                drive = row[1]
                total_length = int(row[2])

                start_range = (range(0, total_length-seq_length, seq_length))
                stop_range = (range(seq_length, total_length, seq_length))

                for frames in zip(start_range, stop_range):
                    self.sequences.append((date, drive, frames))

        self.root_dir = root_dir
        self.augmentation = augmentation

        self.pdf_file = pdf_file
        if pdf_file is not None:
            with open(pdf_file, 'rb') as fp:
                data = pickle.load(fp)
            self.angle_pdf = data['angle_pdf']
            self.offset_pdf = data['offset_pdf']
        else:
            self.angle_pdf = None
            self.offset_pdf = None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):

        # t0 = time.time()

        date = self.sequences[idx][0]
        drive = self.sequences[idx][1]
        frames = self.sequences[idx][2]

        # t1 = time.time()

        dataset = pykitti.raw(self.root_dir, date, drive, frames=range(frames[0], frames[1]))

        # t2 = time.time()

        R_cam_imu = np.matrix(dataset.calib.T_cam2_imu[0:3,0:3])
        K = np.matrix(dataset.calib.P_rect_20[0:3, 0:3])

        G = np.matrix([[0.], [0.], [1.]])

        images = np.zeros((len(dataset), 3, HEIGHT, WIDTH)).astype(np.float32)
        offsets = np.zeros((len(dataset), 1)).astype(np.float32)
        angles = np.zeros((len(dataset), 1)).astype(np.float32)

        # t3 = time.time()

        if self.augmentation:
            rotation = np.random.uniform(-1., 1.)

            shift = (np.random.uniform(-10., 10.), np.random.uniform(-10., 10.), 0)

            rot = -rotation / 180. * np.pi
            Tf = np.matrix([[1, 0, -WIDTH / 2.], [0, 1, -HEIGHT / 2.], [0, 0, 1]])
            Tb = np.matrix([[1, 0, WIDTH / 2.], [0, 1, HEIGHT / 2.], [0, 0, 1]])
            Rt = Tb * np.matrix(
                [[np.cos(rot), -np.sin(rot), -shift[0]], [np.sin(rot), np.cos(rot), -shift[1]], [0, 0, 1]]) * Tf

        for idx, image in enumerate(iter(dataset.rgb)):

            image_width = WIDTH

            pad_w = WIDTH-image[0].width
            pad_h = HEIGHT-image[0].height

            pad_w1 = int(pad_w/2)
            pad_w2 = pad_w - pad_w1
            pad_h1 = int(pad_h/2)
            pad_h2 = pad_h - pad_h1

            padded_image = np.pad(np.array(image[0]), ((pad_h1, pad_h2), (pad_w1, pad_w2), (0,0)), 'edge')

            R_imu = np.matrix(dataset.oxts[idx].T_w_imu[0:3,0:3])
            G_imu = R_imu.T * G
            G_cam = R_cam_imu * G_imu

            h = np.array(K.I.T*G_cam).squeeze()
            # h /= np.linalg.norm(h[0:2])

            if self.augmentation:

                h = np.array(Rt.I.T * np.matrix(h).T).squeeze()

                padded_image = ndimage.interpolation.rotate(padded_image, rotation, reshape=False, mode='nearest')
                padded_image = ndimage.interpolation.shift(padded_image, shift, mode='nearest')


            padded_image = np.transpose(padded_image, [2, 0, 1]).astype(np.float32) / 255.

            hp1 = np.cross(h, np.array([1, 0, 0]))
            # hp1 = np.array([0, h[2], -h[1]])
            hp2 = np.cross(h, np.array([1, 0, -image_width]))

            hp1 /= hp1[2]
            hp2 /= hp2[2]

            mh = (0.5*(hp1[1]+hp2[1])+pad_h1) / HEIGHT - 0.5
            offset = mh

            angle = np.arctan2(h[0], h[1])
            if angle > np.pi/2:
                angle -= np.pi
            elif angle < -np.pi/2:
                angle += np.pi

            images[idx,:,:,:] = padded_image
            offsets[idx] = offset
            angles[idx] = angle


        # try:
        #     images = np.stack(image_list)
        #     offsets = np.expand_dims(np.stack(offset_list).astype(np.float32), -1)
        #     angles = np.expand_dims(np.stack(angle_list).astype(np.float32), -1)
        # except ValueError as err:
        #     print("ValueError: {0}".format(err))
        #     print(date, drive, frames)

        # t4 = time.time()

        # print("kitti times:")
        # print(t1-t0)
        # print(t2-t1)
        # print(t3-t2)
        # print(t4-t3)

        sample = {'images': images, 'offsets': offsets, 'angles': angles}

        return sample


class Cutout(object):
    def __init__(self, length, bias=False):
        self.length = length
        self.central_bias = bias

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        if self.central_bias:
            x = int(np.around(w/4. * np.random.rand(1) + w/2.))
            y = int(np.around(h/4. * np.random.rand(1) + h/2.))

        else:
            y = np.random.randint(h)
            x = np.random.randint(w)

        lx = np.random.randint(1, self.length)
        ly = np.random.randint(1, self.length)

        y1 = np.clip(y - ly // 2, 0, h)
        y2 = np.clip(y + ly // 2, 0, h)
        x1 = np.clip(x - lx // 2, 0, w)
        x2 = np.clip(x + lx // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img



class KittiRawDatasetPP(Dataset):

    def __init__(self, csv_file, root_dir, seq_length, augmentation=True, pdf_file=None, return_info=False, fill_up=True,
                 im_width=WIDTH, im_height=HEIGHT, scale=1., transform=None, random_subsampling=1., get_split_data=False):

        self.seq_length = seq_length
        self.scale = scale
        self.transform = transform

        self.random_subsampling = random_subsampling

        self.seq_length_os = int(self.seq_length * random_subsampling)

        self.get_split_data = get_split_data

        # self.datasets = {}
        #
        # date_folders = glob.glob(root_dir + "/*")
        # for date_folder in date_folders:
        #     date = os.path.basename(date_folder)
        #     drive_folders = glob.glob(date_folder + "/*")
        #     date_set = {}
        #     for drive_folder in drive_folders:
        #         drive = os.path.basename(drive_folder)
        #         files = glob.glob(drive_folder + "/*.pkl")
        #         files.sort()
        #         date_set[drive] = files
        #     self.datasets[date] = date_set

        self.sequences = []

        with open(csv_file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                date = row[0]
                drive = row[1]
                total_length = int(row[2])
                start_frame = int(row[3])

                if total_length <= self.seq_length_os:
                    self.sequences.append((date, drive, (start_frame, start_frame+total_length)))
                else:

                    start_range = (range(start_frame, start_frame+total_length-self.seq_length_os+1, self.seq_length_os))
                    stop_range = (range(start_frame+self.seq_length_os, start_frame+total_length+1, self.seq_length_os))

                    for frames in zip(start_range, stop_range):
                        self.sequences.append((date, drive, frames))

                    trailing = total_length % self.seq_length_os
                    if trailing > 0:
                        self.sequences.append((date, drive, (start_frame+total_length-trailing, start_frame+total_length)))

        self.root_dir = root_dir
        self.augmentation = augmentation
        self.return_info = return_info

        self.pdf_file = pdf_file
        if pdf_file is not None:
            with open(pdf_file, 'rb') as fp:
                data = pickle.load(fp)
            self.angle_pdf = data['angle_pdf']
            self.offset_pdf = data['offset_pdf']
        else:
            self.angle_pdf = None
            self.offset_pdf = None

        self.fill_up = fill_up
        self.im_width = im_width
        self.im_height = im_height

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):

        # t0 = time.time()

        date = self.sequences[idx][0]
        drive = self.sequences[idx][1]
        frames = self.sequences[idx][2]

        frame_list = list(range(frames[0],frames[1]))

        if self.seq_length != self.seq_length_os and len(frame_list) > self.seq_length:
            frame_list = random.sample(frame_list, self.seq_length)
            frame_list.sort()

        dataset = [self.root_dir + "/" + date + "/" + drive + "/%06d.pkl" % idx for idx in frame_list]

        # print("frames: ", frames)

        if self.fill_up:
            seq_length = self.seq_length
        else:
            seq_length = len(dataset)

        images = np.zeros((seq_length, 3, self.im_height, self.im_width)).astype(np.float32)
        offsets = np.zeros((seq_length, 1)).astype(np.float32)
        angles = np.zeros((seq_length, 1)).astype(np.float32)
        offsets_ema = np.zeros((seq_length, 1)).astype(np.float32)
        angles_ema = np.zeros((seq_length, 1)).astype(np.float32)
        Gs = np.zeros((seq_length, 3)).astype(np.float32)

        # t3 = time.time()

        if self.augmentation:
            rotation = np.random.uniform(-2., 2.)
            # rotation = np.random.uniform(-3., 3.)
            max_shift = 20. * self.scale
            # max_shift = 30. * self.scale

            shift = (np.random.uniform(-max_shift, max_shift), np.random.uniform(-max_shift, max_shift), 0)
            # shift = (10, 100, 0)

            rot = -rotation / 180. * np.pi
            Tf = np.matrix([[1, 0, -self.im_width / 2.], [0, 1, -self.im_height / 2.], [0, 0, 1]])
            Tb = np.matrix([[1, 0, self.im_width / 2.], [0, 1, self.im_height / 2.], [0, 0, 1]])
            Rt = Tb * np.matrix(
                [[np.cos(rot), -np.sin(rot), -shift[0]], [np.sin(rot), np.cos(rot), -shift[1]], [0, 0, 1]]) * Tf

        for i, filename in enumerate(dataset):

            with open(filename, 'rb') as fp:
                data = pickle.load(fp)

            image = np.transpose(data['image'], [1, 2, 0])

            image_width = image.shape[1]

            h = data['horizon_hom']
            h_ema = data['horizon_hom_ema'] if self.get_split_data else h

            if self.return_info:
                Gs[i,:] = data['G'].squeeze()

            if self.augmentation:

                h = np.array(Rt.I.T * np.matrix(h).T).squeeze()
                h_ema = np.array(Rt.I.T * np.matrix(h_ema).T).squeeze()

                angle = np.arctan2(h[0], h[1])
                if angle > np.pi / 2:
                    angle -= np.pi
                elif angle < -np.pi / 2:
                    angle += np.pi

                angle_ema = np.arctan2(h_ema[0], h_ema[1])
                if angle_ema > np.pi / 2:
                    angle_ema -= np.pi
                elif angle_ema < -np.pi / 2:
                    angle_ema += np.pi

                # offset = data['offset']
                # angle = data['angle']

                M = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), rotation, 1)
                M[0,2] += shift[0]
                M[1,2] += -shift[1]
                image = cv2.warpAffine(image, M, (0, 0), borderMode=cv2.BORDER_REPLICATE)

                if self.augmentation and np.random.uniform(0., 1.) > 0.5:
                    image = cv2.flip(image, 1)
                    angle *= -1
                    angle_ema *= -1

                # image = ndimage.interpolation.rotate(image, rotation, reshape=False, mode='nearest')
                # image = ndimage.interpolation.shift(image, shift, mode='nearest')

            else:
                offset = data['offset']
                angle = data['angle']

                offset_ema = data['offset_ema'] if self.get_split_data else offset
                angle_ema = data['angle_ema'] if self.get_split_data else angle

            hp1 = np.cross(h, np.array([1, 0, 0]))
            hp2 = np.cross(h, np.array([1, 0, -image_width]))
            hp1 /= hp1[2]
            hp2 /= hp2[2]

            offset = (0.5 * (hp1[1] + hp2[1])) / self.im_height - 0.5

            hp1 = np.cross(h_ema, np.array([1, 0, 0]))
            hp2 = np.cross(h_ema, np.array([1, 0, -image_width]))
            hp1 /= hp1[2]
            hp2 /= hp2[2]

            offset_ema = (0.5 * (hp1[1] + hp2[1])) / self.im_height - 0.5

            # image = np.transpose(image, [2, 0, 1])

            if self.transform is not None:
                image = self.transform(Image.fromarray((image*255.).astype('uint8')))
            else:
                image = np.transpose(image, [2, 0, 1])

            images[i,:,:,:] = image
            offsets[i] = offset
            angles[i] = angle

            offsets_ema[i] = offset_ema
            angles_ema[i] = angle_ema

        if self.fill_up:
            start = len(dataset)
            for i in range(start, self.seq_length):
                images[i,:,:,:] = images[i-1,:,:,:]
                offsets[i] = offsets[i-1]
                angles[i] = angles[i-1]

        # try:
        #     images = np.stack(image_list)
        #     offsets = np.expand_dims(np.stack(offset_list).astype(np.float32), -1)
        #     angles = np.expand_dims(np.stack(angle_list).astype(np.float32), -1)
        # except ValueError as err:
        #     print("ValueError: {0}".format(err))
        #     print(date, drive, frames)

        # t4 = time.time()

        # print("kitti times:")
        # print(t1-t0)
        # print(t2-t1)
        # print(t3-t2)
        # print(t4-t3)

        sample = {'images': images, 'offsets': offsets, 'angles': angles}

        if self.get_split_data:
            sample['offsets_ema'] = offset_ema
            sample['angles_ema'] = angles_ema

        if self.return_info:
            sample['date'] = date
            sample['drive'] = drive
            sample['start'] = frames[0]
            sample['K'] = np.array(data['K'])
            sample['scale'] = data['scale']
            sample['padding'] = data['padding']
            sample['G'] = Gs

        return sample


if __name__ == "__main__":
    # import matplotlib.pyplot as plt

    # dataset = KittiRawDatasetPP("/tnt/home/kluger/tmp/kitti_split/train.csv", '/home/kluger/tmp/athene/kluger/tmp/kitti/horizons', 1)
    dataset = KittiRawDatasetPP("/home/kluger/tmp/kitti_split/train.csv", '/phys/intern/kluger/tmp/kitti/horizons', 2, augmentation=True)
    # dataset = KittiRawDataset("/tnt/home/kluger/tmp/kitti_split/train.csv", '/tnt/data/scene_understanding/KITTI/rawdata', 1)
    # dataset = KittiRawDataset("/home/kluger/tmp/kitti_split/train.csv", '/phys/intern/kluger/tmp/kitti/rawdata', 2, augmentation=False)

    print("dataset size: ", len(dataset))

    offsets = []
    angles = []

    tic = time.time()

    for idx, sample in enumerate(dataset):
        offsets.append(sample['offsets'][0])
        angles.append(sample['angles'][0])
        if (idx+1) % 1000 == 0:
            print(idx+1)
            break

    toc = time.time()

    print(toc-tic)
    exit(0)

    data = {'offsets': offsets, 'angles': angles}

    # pickle_file = "/home/kluger/tmp/kitti_staistics.pkl"
    #
    # with open(pickle_file, 'wb') as f:
    #     pickle.dump(data, f, -1)
    #
    print("offsets: ", np.min(offsets), np.max(offsets))
    print("angles: ", np.min(angles), np.max(angles))

    images = sample['images']
    image = images[0, :, :, :].transpose((1, 2, 0))
    width = image.shape[1]
    height = image.shape[0]

    offset = offsets[0].squeeze()
    offset += 0.5
    offset *= height
    angle = angles[0].squeeze()


    true_mp = np.array([width / 2., offset])
    true_nv = np.array([np.sin(angle), np.cos(angle)])
    true_hl = np.array([true_nv[0], true_nv[1], -np.dot(true_nv, true_mp)])
    true_h1 = np.cross(true_hl, np.array([1, 0, 0]))
    true_h2 = np.cross(true_hl, np.array([1, 0, -width]))
    true_h1 /= true_h1[2]
    true_h2 /= true_h2[2]

    #
    plt.figure()
    plt.imshow(image)
    plt.plot([true_h1[0], true_h2[0]], [true_h1[1], true_h2[1]], 'g-')

    plt.show()