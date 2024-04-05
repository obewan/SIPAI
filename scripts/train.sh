#!/bin/sh

# cript to test the TrainingMonitored on a release, for local testing
cd ../build
./sipai --en test.json --tf test/images-test1.csv --isx 64 --isy 64 --hsx 64 --hsy 64 --osx 64 --osy 64 --hl 1 --haf ReLU --oaf ReLU --lr 0.005 -m TrainingMonitored
