#!/bin/bash
set -e



#for i in 16 32 64 128
for i in 64
do
#    for domain in  'MNISTToUSPS' 'USPSToMNIST'
    for domain in  'MNISTToUSPS'
    do
        CUDA_VISIBLE_DEVICES=1 python -u main.py --nbit $i --dataset Digits --domain $domain --lamda1 0.01 --lamda2 0.01 --lamda3 0.01

        cd matlab &&
        matlab -nojvm -nodesktop -r "demo_eval_PWCF($i, '$domain', 'Digits', 'T003_bits'); quit;" &&
        cd ..

    done
done
