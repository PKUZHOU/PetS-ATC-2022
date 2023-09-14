#!/bin/bash

model=bert_base

CUDA_VISIBLE_DEVICES=0 python python_scripts/pet_experiments.py --exp_name batching_strategy  --model $model --num_streams 1 --log_dir exp_results/pets/$model --alpha_table_path python_scripts/perf_model/alpha_table_1080ti.dat --beta_table_path python_scripts/perf_model/beta_table_1080ti.dat

