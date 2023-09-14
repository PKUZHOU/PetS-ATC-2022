import torch
# from torch._C import _load_for_lite_interpreter
from transformers.modeling_bert import BertModel, BertConfig
import numpy
import turbo_transformers
import sys
import os
import time
from turbo_transformers.layers import PET_Types, PETBertModel, PETBertConfig

class PET_Server:
    def __init__(self) -> None:
        self.base_torch_model = None
        self.base_tt_model = None
        self.test_device = torch.device('cuda:0')
        self.cfg = BertConfig(num_hidden_layers=12)

        self.task_torch_models = []

    def load_base_torch_model(self):
        self.base_torch_model = BertModel(self.cfg)
        self.base_torch_model.eval()
        if torch.cuda.is_available():
            self.base_torch_model.to(self.test_device)

    def load_shared_w(self):
        base_turbo_model = turbo_transformers.SharedBertModel.from_torch(
            self.base_torch_model, self.test_device, "turbo")
        self.base_tt_model = base_turbo_model
        turbo_transformers.set_num_threads(4)    

    def load_new_task(self, pet_type, model_path = None):
        """
        Load a PET task
        """
        pet_bert_config = PETBertConfig(pet_type = pet_type, num_hidden_layers=12)
        pet_bert_model = PETBertModel(pet_bert_config)
        pet_bert_model.eval()

        # load the shared parts from base torch model
        for k,v in self.base_torch_model.named_parameters():
            if pet_type == PET_Types.bitfit:
                # exclude the bias and layer norm params
                if ("bias" in k) or ("LayerNorm" in k):
                    continue
                pet_bert_model.state_dict()[k].copy_(v.clone())
            elif pet_type == PET_Types.adapters:
                pet_bert_model.state_dict()[k].copy_(v.clone())
            else:
                raise NotImplementedError

        if torch.cuda.is_available():
                pet_bert_model.to(self.test_device)
        
        self.task_torch_models.append(pet_bert_model)
        self.base_tt_model.load_new_task_from_torch(pet_bert_model)

    def init(self):
        print("Init shared Model:")
        self.load_base_torch_model()
        self.load_shared_w()

    def prepare_inputs(self):
        self.batch_size = 4
        #self.batch_size = 1
        self.seq_len = 32
        task_ids = torch.LongTensor([0,0,0,0])
        # task_ids = torch.LongTensor([0,1,2,3])
        #task_ids = torch.LongTensor([0])
        n_samples = torch.LongTensor([1,1,1,1])
        input_ids = torch.randint(low=0,
                                  high=self.cfg.vocab_size - 1,
                                  size=(self.batch_size, self.seq_len),
                                  dtype=torch.long,
                                  device=self.test_device)

        return [input_ids, task_ids, n_samples]


    def run(self, inputs):

        pet_output = self.base_tt_model(inputs[0], task_ids = inputs[1], n_samples = inputs[2])
        print("PET_OUTPUT:",pet_output)
        torch_output = self.task_torch_models[0](inputs[0])
        print("TORCH_OUTPUT", torch_output)

        print((abs(pet_output[0]-torch_output[0])).sum()/abs(pet_output[0]).sum())


if __name__ == '__main__':
    server = PET_Server()
    server.init()
    # server.load_new_task(PET_Types.adapters)
    # server.load_new_task(PET_Types.maskbert)
    # server.load_new_task(PET_Types.diff_pruning)
    server.load_new_task(PET_Types.bitfit)

    # generate random inputs
    inputs = server.prepare_inputs()
    # run the tasks and validate the results 
    server.run(inputs)
