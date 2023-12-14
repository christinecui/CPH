#!/bin/bash
set -e

#for i in 16 32 64 128
for i in 64
do
#    for domain in  'AmazonToDslr' 'AmazonToWebcam' 'DslrToAmazon' 'DslrToWebcam' 'WebcamToAmazon' 'WebcamToDslr'
    for domain in  'AmazonToDslr'
    do
        CUDA_VISIBLE_DEVICES=1 python -u main.py --nbit $i --dataset Office-31 --domain $domain --lamda1 0.1 --lamda2 0.01 --lamda3 0.1
        cd matlab &&
        matlab -nojvm -nodesktop -r "demo_eval_PWCF($i, '$domain', 'Office-31', 'T002_test'); quit;" &&
        cd ..
    done
done
