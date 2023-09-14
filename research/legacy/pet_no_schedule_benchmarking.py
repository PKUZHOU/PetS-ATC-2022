import torch
# from torch._C import _load_for_lite_interpreter
from transformers.modeling_bert import BertModel, BertConfig
import numpy
import turbo_transformers
import sys
import os
import time
import tqdm
from turbo_transformers.layers import PET_Types, PETBertModel, PETBertConfig

class Shadow_Server:
    def __init__(self, model_type = "bert_large") -> None:
        self.torch_model = None
        self.base_tt_model = None
        self.test_device = torch.device('cuda:0')

        self.model_type = model_type
        if model_type == 'distilbert':
            model_config = BertConfig(layers = 6)
        elif model_type == 'bert_large':
            model_config = BertConfig(layers=24, hidden_size = 1024, intermediate_size=4096, num_attention_heads=16)
        elif model_type == 'bert_base':
            model_config = BertConfig()
        self.cfg = model_config
    
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
        del self.torch_model
    
    def load_new_task(self, pet_type, model_path = None):
        """
        Load shadows 
        """
        if self.model_type == 'distilbert':
            pet_bert_config = PETBertConfig(layers = 6, pet_type = pet_type)
        elif self.model_type == 'bert_large':
            pet_bert_config = PETBertConfig(layers=24, hidden_size = 1024, intermediate_size=4096, num_attention_heads=16, pet_type = pet_type)
        elif self.model_type == 'bert_base':
            pet_bert_config = PETBertConfig(pet_type = pet_type)

        pet_bert_model = PETBertModel(pet_bert_config)
        if torch.cuda.is_available():
            pet_bert_model.to(self.test_device)
        mem_before_load = turbo_transformers.get_gpu_mem_usage()
        self.base_tt_model.load_new_task_from_torch(pet_bert_model)
        mem_after_load = turbo_transformers.get_gpu_mem_usage()
        print("GPU memory usage for PET: {} MB".format(mem_after_load - mem_before_load))

    def init(self):
        self.load_torch_model()
        self.load_shared_w()

    def prepare_inputs(self, n_queries = 1024, bs_per_task = 1, seq_len = 128, n_tasks = 4):
        batches = []
        
        self.bs_per_task = bs_per_task
        #self.batch_size = 1
        self.seq_len = seq_len
        self.n_tasks = n_tasks

        self.batch_size = bs_per_task * n_tasks

        for i in range(n_queries // self.batch_size):
            task_ids = torch.arange(0, n_tasks).long()
            n_samples = torch.ones(n_tasks).long() * self.bs_per_task
            input_ids = torch.randint(low=0,
                                      high=self.cfg.vocab_size - 1,
                                      size=(self.batch_size, self.seq_len),
                                      dtype=torch.long,
                                      device=self.test_device)

            batch = [input_ids, task_ids, n_samples]
            batches.append(batch)

        return batches
        
    def run(self, batches, iterations = 100):
        # Warmup
        for i in range(5):
            for batch in batches:
                self.base_tt_model(batch[0], task_ids = batch[1], n_samples = batch[2])

        start = time.time()
        for i in range(iterations):
            for batch in batches:
                self.base_tt_model(batch[0], task_ids = batch[1], n_samples = batch[2])
        elasp_time = time.time() - start
        average_time = elasp_time / iterations
        QPS = self.batch_size * len(batches) / (average_time)
        print("Average time : {}".format(average_time))
        print("QPS : {}".format(QPS))
        with open("sequential_shadow_QPS_task.log", "a+") as f:
            f.write("{}, {}, {}, {}\n".format(self.n_tasks, self.bs_per_task, self.seq_len, QPS))

    def prepare_tasks(self, n_tasks):
        print("Loading {} tasks...".format(n_tasks))
        for i in tqdm.tqdm(range(n_tasks)):
           # server.load_new_task(PET_Types.adapters)
           # server.load_new_task(PET_Types.maskbert)
           # server.load_new_task(PET_Types.diff_pruning)
           # server.load_new_task(PET_Types.bitfit)
            server.load_new_task(i % 4)
        print("Done.")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python " + sys.argv[0] + " n_tasks")
        quit()
    n_tasks = int(sys.argv[1])
    server = Shadow_Server()
    server.init()
    mem_before_load = turbo_transformers.get_gpu_mem_usage()
    server.prepare_tasks(n_tasks)
    mem_after_load = turbo_transformers.get_gpu_mem_usage()
    print("GPU memory usage for weight: {} MB".format(mem_after_load - mem_before_load))
    for bs_per_task in [1]:
        for seq_len in [128]:
            #n_queries = 1024
            n_queries = bs_per_task * n_tasks
            inputs = server.prepare_inputs(n_queries, bs_per_task, seq_len, n_tasks)
            print("Start running for case {%d, %d, %d}..." % (n_tasks, bs_per_task, seq_len))
            server.run(inputs, iterations = 10)
            print("Stop running.")
