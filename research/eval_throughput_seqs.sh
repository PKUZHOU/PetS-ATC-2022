#!/bin/bash
model=bert_base

rm -r exp_results/seqs/$model

for bs_sl in {"4,16","2,32","1,64","1,128"}; do
    bs=${bs_sl: 0: 1}
    sl=${bs_sl: 2 }
    echo $bs
    echo $sl
    for task in {2,4,8,16}; do
	    python python_scripts/sequential_model_serving.py --num_tasks $task --batch $bs --min_seq_length $sl --max_seq_length $sl --model $model 
    done
done