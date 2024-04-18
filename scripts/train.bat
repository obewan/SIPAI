@echo off
REM script to test the TrainingMonitored on a release, for local testing
set res_i=60
set res_h=80
set res_o=100
set hl=1
set algo=PRELU
set lr=0.001
set alrf=0.5
set split=4
set factor=4
set input_folder="E:\Documents\project\sipai_images"
set haa=0.001
set oaa=0.001
set temin=-1000
set temax=1000
pushd ..\out\build\x64-Release\
sipai.exe --en test.json --tfo %input_folder% --isx %res_i% --isy %res_i% --hsx %res_h% --hsy %res_h% --osx %res_o% --osy %res_o% --hl %hl% --haf %algo% --oaf %algo% ^
       --haa %haa% --oaa %oaa% --lr %lr% --is %split% --trf %factor% --eas 10 --rl --bl -m TrainingMonitored -V --par --alr --alrf %alrf% --temin %temin% --temax %temax%
popd
