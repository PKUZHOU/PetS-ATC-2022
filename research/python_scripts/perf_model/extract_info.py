import sys
import os
import argparse

def analysis(cfg):

    save_f = open(cfg.save_path,"w")
    assert(os.path.exists(cfg.dumped_file_path))
    with open(cfg.dumped_file_path,'r') as f:
        lines = f.readlines()
        total_shadow_time = 0
        counter = 0
        for line in lines:
            if "PET_TYPE" in line:  #found a new sample
                segments = line.split(',')
                param = []
                for seg in segments:
                    param.append(int(seg.split(":")[1]))
                pet_type, batch, seq_len = param
                save_f.write("{} {} {} ".format(pet_type, batch, seq_len))
                total_shadow_time = 0

            if cfg.mode == 'beta':    
                if "compute_dense_shadow" in line or "compute_intermediate_shadow" in line\
                    or "compute_output_shadow" in line or "compute_qkv_shadow" in line :
                    if "prepare" in line:
                        continue
                    segments = line.split(" ")
                    op_name, op_time = segments[0][:-1], float(segments[1])
                    total_shadow_time += op_time
                    counter += 1
                    if counter == 4:
                        save_f.write("{}\n".format(total_shadow_time))
                        counter = 0
            elif cfg.mode == 'alpha':
                if "Average time" in line:
                    avg_time = float(line.split(" ")[3])
                    save_f.write("{}\n".format(avg_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dumped_file_path",
        type=str,
        default="beta_perf.log",
        help="path of the dumped file",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default='alpha_orig.dat',
    )

    parser.add_argument(
        "--mode",
        type=str,
        default='alpha',
        choices=['alpha','beta']
    )

    cfg = parser.parse_args()

    analysis(cfg)