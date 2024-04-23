#!/bin/sh

# script to test the TrainingMonitored on a release, for local testing
res_i=50
res_h=50
res_o=50
hl=1
algo="PRELU"
lr=0.0001
alrf=0.5
split=4
factor=4
input_folder=/mnt/e/Documents/project/sipai_images/
haa=0.001
oaa=0.001
temin=-1000
temax=1000
cd ../build
./sipai --en test.json --tfo $input_folder --isx $res_i --isy $res_i --hsx $res_h --hsy $res_h --osx $res_o --osy $res_o --hl $hl --haf $algo --oaf $algo \
       	--haa $haa --oaa $oaa --lr $lr --is $split --trf $factor --eas 10 --rl --bl -m TrainingMonitored -V --par --alr --alrf $alrf --temin $temin --temax $temax --vulkan
