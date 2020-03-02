# Temporally Consistent Horizon Lines

If you use this code, please cite [our paper](https://arxiv.org/abs/1907.10014):
```
@article{kluger2019temporally,
  title={Temporally Consistent Horizon Lines},
  author={Kluger, Florian and Ackermann, Hanno and Yang, Michael Ying and Rosenhahn, Bodo},
  journal={arXiv preprint arXiv:1907.10014},
  year={2019}
}
```

## Prerequisites

TODO

## Pre-trained Models

TODO

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
