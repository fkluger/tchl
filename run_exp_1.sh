#!/bin/bash
echo $PATH

for i in 5 6 7 8 9 10
do
	python resnet/train.py --finetune --epochs 160 --seqlength 1 --batch 128 --cutout --gpu 1 --max_error_loss --downscale 2 --workers 4 --lossmax l1 --seqlength_val 1 --batch_val 256  --seed $i ;
done
