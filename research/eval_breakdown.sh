#!/bin/bash
model=bert_base
mkdir -p exp_results/pets/$model

echo "Running Profiling"

CUDA_VISIBLE_DEVICES=0 python python_scripts/pet_experiments.py --exp_name breakdown  --model $model  > exp_results/pets/$model/execution_time_breakdown.log 2>&1


python python_scripts/calc_breakdown.py --model $model --results_path /workspace/research/exp_results
