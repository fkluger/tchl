#!/bin/bash
echo $PATH

for i in 6 7 8 9 10
do
	python resnet/train.py --finetune --epochs 160 --seqlength 32 --batch 4 --cutout --gpu 0 --max_error_loss --downscale 2 --workers 4 --lossmax l1 --seqlength_val 512 --batch_val 1 --lstm_state_reduction 4 --convlstm --skip --lstm_mem 1 --seed $i ;
done
