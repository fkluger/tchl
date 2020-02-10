#!/bin/bash
echo $PATH
python resnet/train.py --finetune --epochs 160 --seqlength 32 --batch 4 --cutout --gpu 6 --max_error_loss --lstm_state_reduction 4 --downscale 2 --workers 4 --convlstm --lossmax sqrt --skip ;
python resnet/train.py --finetune --epochs 200 --seqlength 32 --batch 4 --cutout --gpu 6 --max_error_loss --lstm_state_reduction 4 --downscale 2 --workers 4 --convlstm --lossmax sqrt --skip ;
python resnet/train.py --finetune --epochs 128 --seqlength 32 --batch 4 --cutout --gpu 6 --max_error_loss --lstm_state_reduction 4 --downscale 2 --workers 4 --convlstm --lossmax sqrt --skip ;
python resnet/train.py --finetune --epochs 160 --seqlength 32 --batch 4 --cutout --gpu 6 --max_error_loss --lstm_state_reduction 4 --downscale 2 --workers 4 --convlstm  --skip ;
python resnet/train.py --finetune --epochs 160 --seqlength 32 --batch 4 --cutout --gpu 6 --max_error_loss --lstm_state_reduction 4 --downscale 2 --workers 4 --convlstm --lossmax sqrt --skip --lr_reduction 0.001;
python resnet/train.py --finetune --epochs 160 --seqlength 32 --batch 4  --gpu 6 --max_error_loss --lstm_state_reduction 4 --downscale 2 --workers 4 --convlstm --lossmax sqrt --skip ;
python resnet/train.py --finetune --epochs 160 --seqlength 32 --batch 4 --cutout --gpu 6 --max_error_loss --lstm_state_reduction 4 --downscale 2 --workers 4 --convlstm --lossmax sqrt --skip --trainable_lstm_init ;
python resnet/train.py --finetune --epochs 160 --seqlength 32 --batch 4 --cutout --gpu 6 --max_error_loss --lstm_state_reduction 2 --downscale 2 --workers 4 --convlstm --lossmax sqrt --skip ;
python resnet/train.py --finetune --epochs 160 --seqlength 32 --batch 4 --cutout --gpu 6 --max_error_loss --lstm_state_reduction 4 --downscale 2 --workers 4 --convlstm --lossmax sqrt  ;
python resnet/train.py --finetune --epochs 160 --seqlength 32 --batch 4 --cutout --gpu 6 --max_error_loss --lstm_state_reduction 4 --downscale 2 --workers 4 --convlstm --lossmax sqrt --bias ;
python resnet/train.py --finetune --epochs 80 --seqlength 32 --batch 4 --cutout --gpu 6 --max_error_loss --lstm_state_reduction 4 --downscale 2 --workers 4 --convlstm --lossmax sqrt --skip --overlap 16;
