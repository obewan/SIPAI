#!/bin/sh

#script to test the TrainingMonitored on a release, for local testing
cd ../build
./sipai --en test.json --tf test/images-test1.csv --isx 56 --isy 56 --hsx 56 --hsy 56 --osx 56 --osy 56 --hl 2 --haf LReLU --oaf LReLU -m TrainingMonitored
