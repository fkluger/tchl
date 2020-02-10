#!/bin/bash
echo $PATH

python resnet_3d/train.py --epochs 160 --seqlength 10 --batch 8 --cutout --gpu 0 --max_error_loss --seqlength_val 128 --net resnet18_3_2d_1_3d --workers 6 --cutout 512 --lb1 BB13 --lb2 BB33 ; 
python resnet_3d/train.py --epochs 160 --seqlength 12 --batch 8 --cutout --gpu 0 --max_error_loss --seqlength_val 128 --net resnet18_3_2d_1_3d --workers 6 --cutout 512 --lb1 BB13 --lb2 BB13 ;
python resnet_3d/train.py --epochs 160 --seqlength 8 --batch 8 --cutout --gpu 0 --max_error_loss --seqlength_val 128 --net resnet18_3_2d_1_3d --workers 6  --cutout 512 --lb1 BB33 --lb2 BB33 ;
python resnet_3d/train.py --epochs 160 --seqlength 8 --batch 8 --cutout --gpu 0 --max_error_loss --seqlength_val 128 --net resnet18_3_2d_1_3d --workers 6  --cutout 512 --lb1 BB13 --lb2 BB35 ;
python resnet_3d/train.py --epochs 160 --seqlength 6 --batch 8 --cutout --gpu 0 --max_error_loss --seqlength_val 128 --net resnet18_3_2d_1_3d --workers 6  --cutout 512 --lb1 BB33 --lb2 BB35 ;
python resnet_3d/train.py --epochs 160 --seqlength 20 --batch 4 --cutout --gpu 0 --max_error_loss --seqlength_val 128 --net resnet18_3_2d_1_3d --workers 6 --cutout 512 --lb1 BB33 --lb2 BB55 ;
python resnet_3d/train.py --epochs 160 --seqlength 18 --batch 4 --cutout --gpu 0 --max_error_loss --seqlength_val 128 --net resnet18_3_2d_1_3d --workers 6 --cutout 512 --lb1 BB35 --lb2 BB55 ;
python resnet_3d/train.py --epochs 160 --seqlength 16 --batch 4 --cutout --gpu 0 --max_error_loss --seqlength_val 128 --net resnet18_3_2d_1_3d --workers 6 --cutout 512 --lb1 BB55 --lb2 BB55 ;

#python resnet_3d/train.py --epochs 160 --seqlength 10 --batch 8 --cutout --gpu 0 --max_error_loss --seqlength_val 128 --net resnet18_3_2d_1_3d_lstm --workers 6 --lossmax sqrt  ;






