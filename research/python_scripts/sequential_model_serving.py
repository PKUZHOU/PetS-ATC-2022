from numpy import s_
from torch.random import seed
import turbo_transformers
import torch
from typing import Optional
import transformers
import argparse
import random
import time
import os

"""

The SEQS baseline 

Should be run with the original TurboTransformers

run_tt_base_in_docker.sh

"""


class MultiModelServer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.torch_model = None
        self.task_models = {}
        self.task_inputs = {}
        self.num_tasks = 0
        self.test_device = None
        self.bert_cfg = None
        self.initialize()

    def initialize(self):
        # set the random seed
        random.seed(self.cfg.seed)
        torch.random.manual_seed(self.cfg.seed)

        # set test device
        self.test_device = torch.device(self.cfg.test_device)
        print("Test device", self.test_device)
        # prepare the torch model
        self.torch_model = self.get_torch_model()
        print("Torch model loaded")

        # prepare tt_models for diffferent tasks
        num_tasks = self.cfg.num_tasks
        self.num_tasks = num_tasks
        print("Total tasks: {}".format(num_tasks))

        for n in range(num_tasks):
            self.add_task("task:{}".format(n))

    def get_torch_model(self):
        # bert base
        if self.cfg.model == "bert_base":
            cfg = transformers.BertConfig(attention_probs_dropout_prob=0.0,
                                  hidden_dropout_prob=0.0)
        # distillbert
        elif self.cfg.model == "distillbert":
            cfg = transformers.BertConfig(num_hidden_layers=6)

        elif self.cfg.model == "bert_large":
            cfg = transformers.BertConfig(num_hidden_layers = 24, hidden_size = 1024,
                                    intermediate_size=4096,
                                    num_attention_heads=16)
        torch_model = transformers.BertModel(cfg).eval().cuda()
        self.bert_cfg = cfg
        return torch_model

    def add_task(self, task_name):
        assert self.test_device is not None
        assert self.torch_model is not None
        # get the tt_model
        turbo_model = turbo_transformers.BertModel.from_torch(
            device = self.test_device,
            model = self.torch_model)

        self.task_models[task_name] = turbo_model
        self.task_inputs[task_name] = self.generate_inputs()
    
    def generate_inputs(self):
        # batch size = 1 for each task
        input_seq = torch.randint(low=0,
                high=self.bert_cfg.vocab_size - 1,
                size=(self.cfg.batch, random.randint(self.cfg.min_seq_length, self.cfg.max_seq_length)),
                dtype=torch.long,
                device=self.test_device)
        return input_seq

    def serial_bert_inference(self):
        res_list = []
        for task_id, input_seq in self.task_inputs.items():
            task_model = self.task_models[task_id]
            res, _ = task_model(input_seq)
            res_list.append(res)

        return res_list

    def warmup(self):
        for i in range(5):
            input_ids = torch.randint(low=0,
                                  high=self.bert_cfg.vocab_size - 1,
                                  size=(4, 128),
                                  dtype=torch.long,
                                  device=self.test_device)
            for task_id, _ in self.task_inputs.items():
                task_model = self.task_models[task_id]
                res, _ = task_model(input_ids)

    def simulate_serving(self):
        # warmup
        self.warmup()    

        s_res = []
        start = time.time()
        for round in range(self.cfg.test_rounds):
            s_res = self.serial_bert_inference()
        elasp_time = time.time() - start
        average_time = elasp_time / self.cfg.test_rounds
        QPS = self.cfg.batch * self.cfg.num_tasks / average_time
        print("Average time : {}".format(average_time))
        print("QPS: {}".format(QPS))

        log_path = "exp_results/seqs/"+self.cfg.model
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        with open(log_path+"/serving_throughput_SEQS.log", "a+") as f:
            f.write("task_num:{},bs:{},seq_len:{}\nQPS: {}\n".format(self.cfg.num_tasks, self.cfg.batch, self.cfg.min_seq_length, int(QPS)))

def run_exp(cfg):
    server = MultiModelServer(cfg)
    server.simulate_serving()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=2,
        help="Number of concurrent tasks",
    )
    parser.add_argument(
        "--test_device",
        type=str,
        default='cuda:0',
    )
    parser.add_argument(
        "--min_seq_length",
        type=int,
        default= 128,
    )
    parser.add_argument(
        "--max_seq_length",
        type = int,
        default = 128
    )
    parser.add_argument(
        "--batch",
        type = int,
        default = 1
    )
    parser.add_argument(
        "--model",
        type = str,
        default = "distillbert",
        choices=['bert_base', 'distillbert', 'bert_large']
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1
    )
    parser.add_argument(
        "--test_rounds",
        type=int,
        default=10
    )

    cfg = parser.parse_args()
    run_exp(cfg)