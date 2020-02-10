#!/bin/bash
echo $PATH

for i in 9 10
do
	python resnet/train.py --finetune --epochs 160 --seqlength 32 --batch 4 --cutout --gpu 1 --max_error_loss --downscale 2 --workers 4 --lossmax l1 --seqlength_val 512 --batch_val 1 --lstm_state_reduction 4 --convlstm --skip --simple_skip  --seed $i ;
done
