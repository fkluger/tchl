import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.kde import gaussian_kde
from scipy.stats import norm
import glob
import os
import csv
from datasets.kitti import KittiRawDatasetPP, WIDTH, HEIGHT
from datasets.hlw import HLWDataset
import torch

class KittiAnalyser:

    def __init__(self, dataset):

        self.dataset = dataset

        self.loader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=1,
                                                   shuffle=False)

        self.results = []
        self.all_offsets = []
        self.all_angles = []
        self.all_means = []

        self.mean_offset = None
        self.mean_angle = None
        self.median_offset = None
        self.median_angle = None
        self.mean_image = None


    def analyse(self):

        self.results = []
        self.all_offsets = []
        self.all_angles = []
        self.all_means = []


        for idx, sample in enumerate(self.loader):

            image = sample['images'][0,0,:,:,:].numpy().squeeze()
            offset = sample['offsets'][0,0].numpy().squeeze()
            angle = sample['angles'][0,0].numpy().squeeze()

            image_mean = np.mean(np.reshape(image, (3,-1)), axis=-1)

            self.all_offsets.append(offset)
            self.all_angles.append(angle)
            self.all_means.append(image_mean)

            print(idx)
            # if idx == 100:
            #     break

        self.mean_offset = np.mean(self.all_offsets)
        self.mean_angle = np.mean(self.all_angles)
        self.median_offset = np.median(self.all_offsets)
        self.median_angle = np.median(self.all_angles)
        self.mean_image = np.mean(np.vstack(self.all_means), axis=0).squeeze()

    def write_csv(self, filename):
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ')
            for r in self.results:
                writer.writerow([r[0], r[1], "%d" % r[2], "%.6f" % r[3], "%.6f" % r[4], "%.6f" % r[5], "%.6f" % r[6]])


if __name__ == '__main__':


    # root_dir = "/data/kluger/datasets/kitti/horizons"
    root_dir = "/data/scene_understanding/HLW"
    csv_base = "/home/kluger/tmp/kitti_split_3"

    # dataset = KittiRawDatasetPP(root_dir=root_dir, pdf_file=None, augmentation=False,
    #                                   csv_file=csv_base + "/train.csv", seq_length=1, fill_up=False)
    dataset = HLWDataset(root_dir=root_dir, augmentation=False, set='train')

    anal = KittiAnalyser(dataset)

    anal.analyse()

    print("offset, mean: %.6f -- median: %.6f" % (anal.mean_offset, anal.median_offset))
    print("angle, mean: %.6f -- median: %.6f" % (anal.mean_angle, anal.median_angle))
    print("pixel means: %.6f, %.6f %.6f" % (anal.mean_image[0], anal.mean_image[1], anal.mean_image[2]))

    exit(0)

    anal.write_csv('/home/kluger/tmp/kitti_analysis.csv')

    exit(0)

    pickle_file = "/tnt/home/kluger/tmp/kitti_staistics.pkl"

    with open(pickle_file, 'rb') as fp:
        data = pickle.load(fp)

    print('loaded')

    offsets = np.array(data['offsets'])
    angles = np.array(data['angles'])

    # choice = np.random.choice(offsets.shape[0], 100)
    offsets = offsets.transpose()
    angles = angles.transpose()

    o_pdf = gaussian_kde(offsets)
    a_pdf = gaussian_kde(angles)
    x = np.linspace(-1.,1.,1500)

    fig1, ax1a = plt.subplots()

    ax1a.hist(offsets.transpose(), bins=50)

    ax1b = ax1a.twinx()
    pdf_data = o_pdf(x)
    ax1b.set_ylim(bottom=0, top=np.max(pdf_data))
    ax1b.plot(x, pdf_data, '-b')
    ax1b.plot(x, 1/pdf_data, '-b')

    fig2, ax2a = plt.subplots()

    ax2a.hist(angles.transpose(), bins=50)

    ax2b = ax2a.twinx()
    pdf_data = a_pdf(x)
    ax2b.set_ylim(bottom=0, top=np.max(pdf_data))
    ax2b.plot(x, pdf_data, '-g')

    plt.show()

    pickle_file = "/tnt/home/kluger/tmp/kitti_split/data_pdfs.pkl"
    data = {"offset_pdf": o_pdf, "angle_pdf": a_pdf}
    with open(pickle_file, 'wb') as f:
        pickle.dump(data, f, -1)