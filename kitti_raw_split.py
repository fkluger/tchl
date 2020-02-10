"""Example of pykitti.odometry usage."""

import pykitti
import glob
import os
import random
import csv
import numpy as np
random.seed(1)

# Change this to the directory where you store KITTI data
basedir = '/data/scene_understanding/KITTI/rawdata'

target = '/home/kluger/tmp/kitti_split_4/'

dates = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']

exclusion_list = [
    # calibration:
    "2011_09_26_0119",
    "2011_09_28_0225",
    "2011_09_29_0108",
    "2011_09_30_0072",
    "2011_10_03_0058",
    # person:
    # "2011_09_28_0053", keep one
    "2011_09_28_0054",
    "2011_09_28_0057",
    "2011_09_28_0065",
    "2011_09_28_0066",
    "2011_09_28_0068",
    "2011_09_28_0070",
    "2011_09_28_0071",
    "2011_09_28_0075",
    "2011_09_28_0077",
    "2011_09_28_0078",
    "2011_09_28_0080",
    "2011_09_28_0082",
    "2011_09_28_0086",
    "2011_09_28_0087",
    "2011_09_28_0089",
    "2011_09_28_0090",
    "2011_09_28_0094",
    "2011_09_28_0095",
    "2011_09_28_0096",
    "2011_09_28_0098",
    "2011_09_28_0100",
    "2011_09_28_0102",
    "2011_09_28_0103",
    "2011_09_28_0104",
    "2011_09_28_0106",
    "2011_09_28_0108",
    "2011_09_28_0110",
    "2011_09_28_0113",
    "2011_09_28_0117",
    "2011_09_28_0119",
    "2011_09_28_0121",
    "2011_09_28_0122",
    "2011_09_28_0125",
    "2011_09_28_0126",
    "2011_09_28_0128",
    "2011_09_28_0132",
    "2011_09_28_0134",
    "2011_09_28_0135",
    "2011_09_28_0136",
    "2011_09_28_0138",
    "2011_09_28_0141",
    "2011_09_28_0143",
    "2011_09_28_0145",
    "2011_09_28_0146",
    "2011_09_28_0149",
    "2011_09_28_0153",
    "2011_09_28_0154",
    "2011_09_28_0155",
    "2011_09_28_0156",
    "2011_09_28_0160",
    "2011_09_28_0161",
    "2011_09_28_0162",
    "2011_09_28_0165",
    "2011_09_28_0166",
    "2011_09_28_0167",
    "2011_09_28_0168",
    "2011_09_28_0171",
    "2011_09_28_0174",
    "2011_09_28_0177",
    "2011_09_28_0179",
    "2011_09_28_0183",
    "2011_09_28_0184",
    "2011_09_28_0185",
    "2011_09_28_0186",
    "2011_09_28_0187",
    "2011_09_28_0191",
    "2011_09_28_0192",
    "2011_09_28_0195",
    "2011_09_28_0198",
    "2011_09_28_0199",
    "2011_09_28_0201",
    "2011_09_28_0204",
    "2011_09_28_0205",
    "2011_09_28_0208",
    "2011_09_28_0209",
    "2011_09_28_0214",
    "2011_09_28_0216",
    "2011_09_28_0220",
    "2011_09_28_0222"
]

exclusion_list = [
    # calibration:
    "2011_09_26_0119",
    "2011_09_28_0225",
    "2011_09_29_0108",
    "2011_09_30_0072",
    "2011_10_03_0058"]

test_split = 0.15
val_split = 0.15

datasets_w_len = []
total_size = 0
num_drives = 0

split_length = 512

if not os.path.exists(target):
    os.makedirs(target)

for date in dates:

    date_dir = basedir + "/" + date

    drive_dirs = glob.glob(date_dir + "/*sync")
    drive_dirs.sort()

    drives = []
    for drive_dir in drive_dirs:

        drive = drive_dir.split("_")[-2]

        # if (date + "_" + drive) in exclusion_list:
        #     continue

        drives.append(drive)

    for drive in drives:

        print(date, drive)

        dataset = pykitti.raw(basedir, date, drive)

        if len(dataset) > split_length:
            frames_left = len(dataset)
            start_frame = 0
            while frames_left > 0:
                frames_here = np.minimum(frames_left, split_length)
                datasets_w_len.append((date, drive, frames_here, start_frame))
                start_frame += frames_here
                frames_left -= frames_here

        else:
            datasets_w_len.append((date, drive, len(dataset), 0))

        total_size += len(dataset)
        num_drives += 1

print("total size: ", total_size)
print("num_drives: ", num_drives)

exit(0)

max_num_test = int(test_split * total_size)
max_num_val = int(val_split * total_size)

num_test = 0
num_val = 0

test_drives = []
val_drives = []
train_drives = []

max_tries = 10

num_tries = 0
while num_test < max_num_test and num_tries < max_tries:
    choice = random.choice(datasets_w_len)

    if num_test+choice[2] > max_num_test:
        num_tries += 1
        continue
    else:
        test_drives.append(choice)
        num_test += choice[2]
        datasets_w_len.remove(choice)

print("test: ", num_test)

num_tries = 0
while num_val < max_num_val and num_tries < max_tries:
    choice = random.choice(datasets_w_len)

    if num_val+choice[2] > max_num_val:
        num_tries += 1
        continue
    else:
        val_drives.append(choice)
        num_val += choice[2]
        datasets_w_len.remove(choice)

print("val: ", num_val)

print("rest: ", total_size-num_val-num_test)


with open(target + 'test.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    for drive in test_drives:
        writer.writerow([drive[0], drive[1], "%d" % drive[2], "%d" % drive[3]])

with open(target + 'val.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    for drive in val_drives:
        writer.writerow([drive[0], drive[1], "%d" % drive[2], "%d" % drive[3]])

with open(target + 'train.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    for drive in datasets_w_len:
        writer.writerow([drive[0], drive[1], "%d" % drive[2], "%d" % drive[3]])