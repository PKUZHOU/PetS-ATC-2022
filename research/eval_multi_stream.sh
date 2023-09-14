#!/bin/bash
model=bert_base

CUDA_VISIBLE_DEVICES=0 python python_scripts/pet_experiments.py --exp_name multi_stream  --model $model  --log_dir exp_results/pets/$model