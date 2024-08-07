#!/bin/sh

# script to test the Training on a release, for local testing
res_i=8
res_h=9
res_o=10
hl=1
algo="LRELU"
lr=0.001
alrf=0.5
split=4
factor=4
input_folder=test/data/images/target
haa=0.001
oaa=0.001

cd ../build
./sipai --en test.json --tfo $input_folder --isx $res_i --isy $res_i --hsx $res_h --hsy $res_h --osx $res_o --osy $res_o --hl $hl --haf $algo --oaf $algo --haa $haa --oaa $oaa --lr $lr --is $split --trf $factor --eas 10 --rl --bl -m Training -V --par --alr --alrf $alrf --vulkan
