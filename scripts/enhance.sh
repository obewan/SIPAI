#!/bin/sh

#script to test a release enhance mode, for local testing
cd ../build
./sipai --in "test.json" --if "test/data/images/input/001a.png" --of "test/data/images/output/001a_enh.png" --os 2 -m "Enhancer"
