#!/bin/sh

# require gprof2dot (https://github.com/jrfonseca/gprof2dot) 
# and a first run of sipai under Debug target 
# that should have create a gmon.out file.
# Also create a .gprof folder.
# You can check a textual output as well with: 
# gprof sipai gmon.out > ../.gprof/analysis.txt

cd ../build
gprof sipai | gprof2dot | dot -Tpng -o ../.gprof/output_$(date +%Y%m%d_%H%M).png
