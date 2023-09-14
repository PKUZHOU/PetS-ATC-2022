#!/bin/sh

#for i in {1,2,4,8,16,32}; do
for i in 1; do
    echo $i
    CUDA_VISIBLE_DEVICES=0 python pet_no_schedule_benchmarking.py $i
done;
