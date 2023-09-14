#!/bin/bash

#echo "-----------Single batch multi model------------"
#python multi_model_serving.py --num_tasks 8 --batch 1

#echo "-----------Single model multi batch------------"
#python multi_model_serving.py --num_tasks 1 --batch 8

for bs in {8..8}; do
    for sl in {8,16,32,64,128,256}; do
	echo $bs
	echo $sl
	python multi_model_serving.py --num_tasks 1 --batch $bs --min_seq_length $sl --max_seq_length $sl
    done
done
