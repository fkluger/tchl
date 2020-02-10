#!/bin/bash
echo $PATH

python resnet/train.py --finetune --epochs 160 --seqlength 1 --batch 128 --cutout --gpu 6 --max_error_loss --lstm_state_reduction 4 --downscale 2 --workers 4 --convlstm  --skip ;
python resnet/train.py --finetune --epochs 160 --seqlength 32 --batch 4 --cutout --gpu 6 --max_error_loss --lstm_state_reduction 4 --downscale 2 --workers 4 --convlstm ;
python resnet/train.py --finetune --epochs 160 --seqlength 32 --batch 4 --cutout --gpu 6 --max_error_loss --lstm_state_reduction 4 --downscale 2 --workers 4 --convlstm --skip --trainable_lstm_init ;
python resnet/train.py --finetune --epochs 160 --seqlength 32 --batch 4 --cutout --gpu 6 --max_error_loss --lstm_state_reduction 4 --downscale 2 --workers 4 --convlstm --bias ;
python resnet/train.py --finetune --epochs 160 --seqlength 32 --batch 4 --cutout --gpu 6 --max_error_loss --lstm_state_reduction 4 --downscale 2 --workers 4 --convlstm --skip --lr_reduction 0.001;
python resnet/train.py --finetune --epochs 160 --seqlength 32 --batch 4 --cutout --gpu 6 --max_error_loss --lstm_state_reduction 2 --downscale 2 --workers 4 --convlstm --lossmax sqrt --skip ;
python resnet/train.py --finetune --epochs 80 --seqlength 32 --batch 4 --cutout --gpu 6 --max_error_loss --lstm_state_reduction 4 --downscale 2 --workers 4 --convlstm --skip --overlap 16;


