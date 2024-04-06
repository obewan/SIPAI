#!/bin/sh
#script to make a release, for local testing
#for a rebuild, use the script with any parameter
#like ./release.sh rebuild
set -e 

cd ..
if [ "$1" ]; then
 rm -rf ./build
 cmake -B ./build -DCMAKE_BUILD_TYPE=Release
fi
cmake --build ./build --config Release -- -j 2

