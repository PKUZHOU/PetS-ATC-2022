from tqdm import tqdm
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

    def load_torch_model(self):
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
        Load PETs 
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
            elif pet_type == PET_Types.diff_pruning:
                if ("bias" in k) or ("LayerNorm" in k):
                    continue
            pet_bert_model.state_dict()[k].copy_(v.clone())
            
        if torch.cuda.is_available():
                pet_bert_model.to(self.test_device)
        
        self.task_torch_models.append(pet_bert_model)
        self.base_tt_model.load_new_task_from_torch(pet_bert_model)

    def init(self):
        print("Init shared Model:")
        self.load_torch_model()
        self.load_shared_w()

    def prepare_inputs(self):
        self.batch_size = 4
        self.seq_len = 32
        task_ids = torch.LongTensor([0,1,2,3])
        n_samples = torch.LongTensor([1,1,1,1])
        input_ids = torch.randint(low=0,
                                  high=self.cfg.vocab_size - 1,
                                  size=(self.batch_size, self.seq_len),
                                  dtype=torch.long,
                                  device=self.test_device)

        return [input_ids, task_ids, n_samples]

    def run(self, inputs):
        pet_output = self.base_tt_model(inputs[0], task_ids = inputs[1], n_samples = inputs[2])[0]
        print("PET_OUTPUT:",pet_output)

        tc_outputs = []
        for i, tc_model in enumerate(self.task_torch_models):
            torch_output = tc_model(inputs[0][i:i+1])
            tc_outputs.append(torch_output[0])
        
        tc_outputs = torch.cat(tc_outputs,dim=0)
        print("TORCH_OUTPUT", tc_outputs)

        relative_error = float((abs(pet_output-tc_outputs)).sum()/abs(tc_outputs).sum().cpu().data)
        print("Relative Error: {:.4%}".format(relative_error))


if __name__ == '__main__':
    server = PET_Server()
    # load the shared model
    server.init()

    # load for tasks:
    print("Loading PET tasks...")
    for pet_type in tqdm([PET_Types.bitfit, PET_Types.adapters, PET_Types.diff_pruning, PET_Types.maskbert]):
        server.load_new_task(pet_type)

    print("Prepare inputs...")
    inputs = server.prepare_inputs()
    server.run(inputs)
