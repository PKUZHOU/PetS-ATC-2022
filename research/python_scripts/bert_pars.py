import subprocess
import os
import sys
def run(command, num_streams, batch_size, seq_len):
    print("num_streams={0}, batch_size={1}, seq_len={2}".format(num_streams, batch_size, seq_len))
    output = subprocess.getoutput(command)
    print(output)

if __name__ == "__main__":
    project_root = "/src/"
    m_model_path =  "/workspace/research/model/bert-base-uncased.npz"

    if not os.path.exists(m_model_path):
        print("Please download the bert model!")
        exit(1)

    m_num_streams = [2,4,8,16]
    m_batch_size = [1]
    m_seq_len = [128]
    m_num_iters = 10

    binary = project_root + "build/example/cpp/bert_model_multi_stream"
    print("Start Running PARS ...")
    for num_streams in m_num_streams:
        for batch_size in m_batch_size:
            for seq_len in m_seq_len:
                command = binary + " " + m_model_path + " " + str(num_streams) + " " \
                + str(batch_size) + " " + str(seq_len) + " " + str(m_num_iters)
                run(command, num_streams, batch_size, seq_len)
    

