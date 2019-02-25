#!/usr/bin/env bash

python save_CurveModel.py --dir=points2plane/middle_point5051_1 --init_start=curves/curve50/checkpoint-100.pt --init_middle=curves/middle_init5051_1/checkpoint-100.pt --init_end=curves/curve51/checkpoint-100.pt

python plane.py --dir=plots/middle_point5051_1 --data_path=data --model=VGG16 --curve=PolyChain --ckpt=points2plane/middle_point5051_1/checkpoint--1.pt

for i in $(seq 2 6);
do
    prev=$(expr $i - 1)
    
    python save_CurveModel.py --dir=points2plane/middle_point5051_$i --init_start=curves/curve50/checkpoint-100.pt --init_middle=curves/middle_init5051_$i/checkpoint-100.pt --init_end=curves/middle_init5051_$prev/checkpoint-100.pt
    
    python plane.py --dir=plots/middle_point5051_$i --data_path=data --model=VGG16 --curve=PolyChain --ckpt=points2plane/middle_point5051_$i/checkpoint--1.pt
done

