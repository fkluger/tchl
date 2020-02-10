#!/bin/bash
echo $PATH

python resnet/train.py --split 5 --finetune --epochs 160 --seqlength 32 --batch 4 --cutout 512 --gpu 6 --max_error_loss --lstm_state_reduction 2 --downscale 2 --workers 4 --convlstm --skip;
python resnet/train.py --split 5 --finetune --epochs 160 --seqlength 8  --batch 16 --cutout 512 --gpu 6 --max_error_loss --lstm_state_reduction 4 --downscale 2 --workers 4 --convlstm --skip;
python resnet/train.py --split 5 --finetune --epochs 160 --seqlength 4  --batch 32 --cutout 512 --gpu 6 --max_error_loss --lstm_state_reduction 4 --downscale 2 --workers 4 --convlstm --skip;



