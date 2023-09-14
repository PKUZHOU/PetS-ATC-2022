#!/bin/sh

#for batch_size in {1,2,4,8,16,32}; do
#    for seq_len in {1,2,4,6,8,10,12,16,20,24,28,32,36,40,48,56,64,72,80,88,96,128}; do
#	for pet_type in {0,1,2,3}; do
#	    echo $pet_type $batch_size $seq_len
#	    echo $pet_type $batch_size $seq_len >> perf_model2.log
#	    python pet_perf_model_profiling.py $pet_type $batch_size $seq_len > perf_model.log 2>&1
#	    cat perf_model.log | grep "compute_qkv_shadow" >> perf_model2.log
#	done
#    done
#done

for batch_size in {1,2,4,8,16,32}; do
    for seq_len in {1,2,4,6,8,10,12,16,20,24,28,32,36,40,48,56,64,72,80,88,96,128}; do
	echo $batch_size $seq_len
	echo $batch_size $seq_len >> alpha_orig.dat
	python pet_perf_model_profiling.py 0 $batch_size $seq_len > perf_model.log 2>&1
	cat perf_model.log | grep "Average time" >> alpha_orig.dat
    done
done
