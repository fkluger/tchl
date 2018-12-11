from datasets.kitti import KittiRawDatasetPP, WIDTH, HEIGHT
import torch
import numpy as np

num_bins = 10

# root_dir = "/phys/intern/kluger/tmp/kitti/horizons_s0.500"
root_dir = "/tnt/data/kluger/datasets/kitti/horizons_s0.500"
csv_base = "/tnt/home/kluger/tmp/kitti_split_3"

dataset = KittiRawDatasetPP(root_dir=root_dir, csv_file=csv_base + "/train.csv", seq_length=16, fill_up=False,
                            im_height=HEIGHT // 2, im_width=WIDTH // 2, scale=2.)

loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)

all_offsets = []
all_angles = []

for idx, sample in enumerate(loader):

    offsets = sample['offsets'].cpu().numpy()
    angles = sample['angles'].cpu().numpy()

    print("idx %d / %d" % (idx, len(loader)))
    all_offsets = []
    all_offsets_estm = []

    for si in range(offsets.shape[1]):
        all_offsets += [offsets[:,si,:,].squeeze()]
        all_angles += [angles[:,si,:,].squeeze()]

    break

all_angles.sort()
all_offsets.sort()

# all_angles = [-np.inf] + all_angles + [np.inf]
# all_offsets = [-np.inf] + all_offsets + [np.inf]

num_entries = len(all_offsets)

cdf = np.linspace(0., 1., num_entries+1)[1:]
cdf_bins = np.linspace(0., 1., num_bins+1)[1:]

off_edges = []
ang_edges = []

cdf_id = 0
for cdf_bin_id in range(cdf_bins.shape[0]-1):
    while cdf[cdf_id] < cdf_bins[cdf_bin_id]:
        cdf_id += 1
    else:
        w1 = cdf[cdf_id] - cdf_bins[cdf_bin_id]
        w2 = - cdf[cdf_id-1] + cdf_bins[cdf_bin_id]
        w1 /= (w1+w2)
        w2 /= (w1+w2)

        off_upper_edge = all_offsets[cdf_id-1] * w1 + all_offsets[cdf_id] * w2
        ang_upper_edge = all_angles[cdf_id-1] * w1 + all_angles[cdf_id] * w2

        off_edges += [off_upper_edge]
        ang_edges += [ang_upper_edge]

print("done")