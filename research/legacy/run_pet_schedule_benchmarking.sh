#!/bin/bash

for i in {1,2,4,8,16,32}; do
    echo $i
    CUDA_VISIBLE_DEVICES=1 python pet_schedule_benchmarking.py --sort_queries --test_device cuda:0 --num_streams $i --num_tasks 32
done;