import argparse
import os

def calc_breakdown(cfg):
    model = cfg.model
    results_path = cfg.results_path
    dump_file_path = os.path.join(results_path,"pets",model,"execution_time_breakdown.log")
    if not os.path.exists(dump_file_path):
        print(dump_file_path,"does not exist!")
        return
    with open(dump_file_path,'r') as f:
        lines = f.readlines()
        total_shadow_ratio = 0
        counter = 0
        for line in lines:
            if "PET Time line" in line:  #found a new sample
                total_shadow_ratio = 0

            if "compute_dense_shadow" in line or "compute_intermediate_shadow" in line\
                or "compute_output_shadow" in line or "compute_qkv_shadow" in line :
                if "prepare" in line:
                    continue
                segments = line.split(" ")
                op_name, op_ratio = segments[0][:-1], float(segments[3])
                total_shadow_ratio += op_ratio
                counter += 1
                if counter == 4:
                    print("PET Operators Execution Time: {:.2%}".format(total_shadow_ratio/100))
                    total_shadow_ratio = 0
                    counter = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--model",
            type=str,
            default='bert_base',
            choices=['bert_base',"bert_large","distil_bert"]
        )
    parser.add_argument(
        "--results_path",
        type=str,
        default="/workspace/research/exp_results"
    )
    cfg = parser.parse_args()
    calc_breakdown(cfg)