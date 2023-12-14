#!/bin/bash
set -e

#for i in 16 32 64 128
for i in 64
do
#    for domain in 'ArtToReal_World' 'Real_WorldToArt' 'ClipartToReal_World' 'Real_WorldToClipart' 'ProductToReal_World' 'Real_WorldToProduct'
    for domain in 'ProductToReal_World'
    do
        CUDA_VISIBLE_DEVICES=1 python -u main.py --nbit $i --dataset Office-Home --domain $domain --lamda1 1 --lamda2 1 --lamda3 100
        cd matlab &&
        matlab -nojvm -nodesktop -r "demo_eval_PWCF($i, '$domain', 'Office-Home', 'T001'); quit;" &&
        cd ..
    done
done
