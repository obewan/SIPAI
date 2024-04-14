#!/bin/sh

# script to test the TrainingMonitored on a release, for local testing
res_i=60
res_h=65
res_o=70
hl=1
algo="LReLU"
lr=0.0003
split=4
factor=5
input_folder=/mnt/e/Documents/project/sipai_images/

cd ../build
./sipai --en test.json --tfo $input_folder --isx $res_i --isy $res_i --hsx $res_h --hsy $res_h --osx $res_o --osy $res_o --hl $hl --haf $algo --oaf $algo --lr $lr --is $split --trf $factor --eas 200 --rl --bl -m TrainingMonitored
