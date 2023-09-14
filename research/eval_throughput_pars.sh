#!/bin/bash

model=bert_base


echo "PARS running..."

rm -r exp_results/pars/$model
mkdir -p exp_results/pars/$model

CUDA_VISIBLE_DEVICES=0 python python_scripts/bert_pars.py | tee exp_results/pars/bert_base/serving_throughput_PARS.log