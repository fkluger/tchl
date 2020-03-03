# Temporally Consistent Horizon Lines

If you use this code, please cite [our paper](https://arxiv.org/abs/1907.10014):
```
@inproceedings{kluger2020temporally,
  title={Temporally Consistent Horizon Lines},
  author={Kluger, Florian and Ackermann, Hanno and Yang, Michael Ying and Rosenhahn, Bodo},
  booktitle={2020 International Conference on Robotics and Automation (ICRA)},
  year={2020}
}
```

## Setup
Get the code:
```
git clone --recurse-submodules https://github.com/fkluger/tchl.git
cd tchl
git submodule update --init --recursive
```

Set up the Python environment using [Anaconda](https://www.anaconda.com/): 
```
conda env create -f environment.yml
source activate tchl
```

[Download](https://cloud.tnt.uni-hannover.de/index.php/s/YPWqXyD7Hm5c71u) the preprocessed KITTI Horizon data or [generate it yourself](https://github.com/fkluger/kitti_horizon).

## Pre-trained Models

You can download the pre-trained model weights here:
* [Temporally consistent ConvLSTM CNN](https://cloud.tnt.uni-hannover.de/index.php/s/qZaD0cvk9DwGcbk)
* [Single-frame baseline](https://cloud.tnt.uni-hannover.de/index.php/s/BrKMoVXOROhQSPN)
* [Non-temporal (ablation study)](https://cloud.tnt.uni-hannover.de/index.php/s/38Tnb9E1Yh7Ye8q)
* [Naïve residual (ablation study)](https://cloud.tnt.uni-hannover.de/index.php/s/2rOJUtScXfCXvkG)

## Training

In order to train the temporally consistent ConvLSTM network on KITTI Horizon, simply run:
```
python convlstm_net/train.py --convlstm --skip --max_error_loss --dataset_path PATH_TO_PREPROCESSED_DATASET 
```

For the single frame baseline, run:
```
python convlstm_net/train.py --seqlength 1 --batch 128 --max_error_loss --dataset_path PATH_TO_PREPROCESSED_DATASET 
```

## Evaluation
In order to evaluate the temporally consistent CNN on KITTI Horizon, run:
```
python convlstm_net/evaluate.py --whole --skip --convlstm --cpu --load temporally_consistent.ckpt --set test --dataset_path PATH_TO_PREPROCESSED_DATASET
```
For the single-frame baseline, run:
```
python convlstm_net/evaluate.py --whole --cpu --load single_frame.ckpt --set test --dataset_path PATH_TO_PREPROCESSED_DATASET
```

