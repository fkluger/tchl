#!/bin/bash
echo $PATH

python resnet/train.py --finetune --epochs 160 --seqlength 1 --batch 128 --cutout --gpu 5 --max_error_loss --downscale 2 --workers 4 --lossmax sqrt --seqlength_val 1 --batch_val 256 --lstm_state_reduction 4 --convlstm --skip ;
python resnet/train.py --finetune --epochs 200 --seqlength 1 --batch 128 --cutout --gpu 5 --max_error_loss --downscale 2 --workers 4 --lossmax sqrt --seqlength_val 1 --batch_val 256 --lstm_state_reduction 4 --convlstm --skip ;
python resnet/train.py --finetune --epochs 160 --seqlength 1 --batch 128 --cutout --gpu 5 --max_error_loss --downscale 2 --workers 4 --lossmax sqrt --seqlength_val 1 --batch_val 256 --lstm_state_reduction 4 --convlstm --skip ;
python resnet/train.py --finetune --epochs 200 --seqlength 1 --batch 128 --cutout --gpu 5 --max_error_loss --downscale 2 --workers 4 --lossmax sqrt --seqlength_val 1 --batch_val 256 --lstm_state_reduction 4 --convlstm --skip ;
python resnet/train.py --finetune --epochs 160 --seqlength 1 --batch 128 --cutout --gpu 5 --max_error_loss --downscale 2 --workers 4 --lossmax sqrt --seqlength_val 1 --batch_val 256 ;
python resnet/train.py --finetune --epochs 200 --seqlength 1 --batch 128 --cutout --gpu 5 --max_error_loss --downscale 2 --workers 4 --lossmax sqrt --seqlength_val 1 --batch_val 256 ;
python resnet/train.py --finetune --epochs 160 --seqlength 16 --batch 8 --cutout --gpu 5 --max_error_loss --lstm_state_reduction 4 --downscale 2 --workers 4 --convlstm --lossmax sqrt --skip ;
python resnet/train.py --finetune --epochs 160 --seqlength 8 --batch 16 --cutout --gpu 5 --max_error_loss --lstm_state_reduction 4 --downscale 2 --workers 4 --convlstm --lossmax sqrt --skip ;
python resnet/train.py --finetune --epochs 160 --seqlength 4 --batch 32 --cutout --gpu 5 --max_error_loss --lstm_state_reduction 4 --downscale 2 --workers 4 --convlstm --lossmax sqrt --skip ;
python resnet/train.py --finetune --epochs 128 --seqlength 1 --batch 128 --cutout --gpu 5 --max_error_loss --downscale 2 --workers 4 --lossmax sqrt --seqlength_val 1 --batch_val 256 ;


