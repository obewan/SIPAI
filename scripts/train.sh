#!/bin/sh

# script to test the TrainingMonitored on a release, for local testing
res_i=80
res_h=90
res_o=100
hl=1
algo="LReLU"
lr=0.0003
split=4
factor=4
input_folder=/mnt/e/Documents/project/sipai_images/

cd ../build
./sipai --en test.json --tfo $input_folder --isx $res_i --isy $res_i --hsx $res_h --hsy $res_h --osx $res_o --osy $res_o --hl $hl --haf $algo --oaf $algo --lr $lr --is $split --trf $factor --eas 10 --rl --bl -m TrainingMonitored -V --par
