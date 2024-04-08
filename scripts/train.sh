#!/bin/sh

# script to test the TrainingMonitored on a release, for local testing
res_i=60
res_h=65
res_o=70
hl=1
algo="LReLU"
lr=0.0002

cd ../build
./sipai --en test.json --tf test/images-test1.csv --isx $res_i --isy $res_i --hsx $res_h --hsy $res_h --osx $res_o --osy $res_o --hl $hl --haf $algo --oaf $algo --lr $lr --eas 200 --bl -m TrainingMonitored
