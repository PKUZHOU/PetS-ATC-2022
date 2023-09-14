#!/bin/bash

dump_file_name="alpha_perf.log"
table_file_name="alpha_table_v100.dat"
max_batch=64
max_seq_len=128
seq_len_step=4

python3 pet_perf_model_profiling_load_once.py --mode alpha --max_batch $max_batch --max_seq_len $max_seq_len --seq_len_step $seq_len_step > $dump_file_name 2>&1
python3 extract_info.py --dumped_file_path $dump_file_name --save_path $table_file_name --mode alpha