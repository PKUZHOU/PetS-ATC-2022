import torch
from transformers.modeling_bert import BertModel, BertConfig
import numpy
import turbo_transformers
import sys
import os
import time
from turbo_transformers.layers import PET_Types, PETBertModel, PETBertConfig
import argparse

class Shadow_Server:
    def __init__(self) -> None:
        self.torch_model = None
        self.base_tt_model = None
        self.test_device = torch.device('cuda:0')
        self.cfg = BertConfig()
    
    def load_torch_model(self):
        self.torch_model = BertModel(self.cfg)
        self.torch_model.eval()
        if torch.cuda.is_available():
            self.torch_model.to(self.test_device)

    def load_shared_w(self):
        base_turbo_model = turbo_transformers.SharedBertModel.from_torch(
            self.torch_model, self.test_device, "turbo")
        self.base_tt_model = base_turbo_model
        turbo_transformers.set_num_threads(4)
    
    def load_new_task(self, pet_type, model_path = None):
        """
        Load shadows 
        """
        pet_bert_config = PETBertConfig(pet_type = pet_type)
        pet_bert_model = PETBertModel(pet_bert_config)

        if torch.cuda.is_available():
                pet_bert_model.to(self.test_device)
        
        self.base_tt_model.load_new_task_from_torch(pet_bert_model)

    def init(self):
        self.load_torch_model()
        self.load_shared_w()
        pass

    def prepare_inputs(self, task_id, batch_size=1, seq_len=1):
        self.batch_size = batch_size
        self.seq_len = seq_len
        task_ids = torch.tensor([task_id]).long()
        n_samples = torch.ones(1).long() * self.batch_size
        input_ids = torch.randint(low=0,
                                  high=self.cfg.vocab_size - 1,
                                  size=(self.batch_size, self.seq_len),
                                  dtype=torch.long,
                                  device=self.test_device)

        return [input_ids, task_ids, n_samples]
        
    def run(self, inputs, iteration = 100):
        # Warmup
        for i in range(4):
            self.base_tt_model(inputs[0], task_ids = inputs[1], n_samples = inputs[2])

        turbo_transformers.enable_perf("Shadow_Server")

        start = time.time()
        for i in range(iteration):
            self.base_tt_model(inputs[0], task_ids = inputs[1], n_samples = inputs[2])
        elasp_time = time.time() - start
        average_time = elasp_time / iteration
        print("Average time : {} ms".format(average_time * 1000), flush=True)
        print("QPS: {}".format(self.batch_size / (average_time)), flush=True)

        turbo_transformers.print_results()

def run(cfg):
    server = Shadow_Server()
    server.init()
    
    for pet_type in range(0, 4):
        server.load_new_task(pet_type)

    if cfg.mode == 'alpha':
        for batch_size in range(cfg.max_batch, 0, -1):
            for seq_len in range(cfg.max_seq_len, 0, -cfg.seq_len_step):
                print("PET_TYPE:{}, BATCH_SIZE:{}, SEQ_LEN:{}".format(4, batch_size, seq_len), flush=True)
                inputs = server.prepare_inputs(pet_type, batch_size, seq_len)
                server.run(inputs, cfg.eval_rounds)

    elif cfg.mode == 'beta':
        for pet_type in range(0, 4):
            for batch_size in range(cfg.max_batch, 0, -1):
                for seq_len in range(cfg.max_seq_len, 0, -cfg.seq_len_step):
                    print("PET_TYPE:{}, BATCH_SIZE:{}, SEQ_LEN:{}".format(pet_type, batch_size, seq_len), flush=True)
                    inputs = server.prepare_inputs(pet_type, batch_size, seq_len)
                    server.run(inputs, cfg.eval_rounds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default='alpha',
        choices=['alpha','beta']
    )
    parser.add_argument(
        "--max_batch",
        type=int,
        default=32
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=128
    )
    parser.add_argument(
        "--seq_len_step",
        type=int,
        default=4
    )
    parser.add_argument(
        "--eval_rounds",
        type=int,
        default=4
    )
    cfg = parser.parse_args()
    run(cfg)

