#!/bin/sh

#script to make a release, for local testing
cd ..
rm -rf ./build
cmake -B ./build -DCMAKE_BUILD_TYPE=Release
cmake --build ./build --config Release -- -j 2

