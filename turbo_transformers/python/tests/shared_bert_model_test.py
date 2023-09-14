# Copyright (C) 2020 THL A29 Limited, a Tencent company.
# All rights reserved.
# Licensed under the BSD 3-Clause License (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at
# https://opensource.org/licenses/BSD-3-Clause
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
# See the AUTHORS file for names of contributors.

import unittest
import torch
from transformers.modeling_bert import BertModel, BertConfig
import numpy
import turbo_transformers
import sys
import os

from turbo_transformers.layers import MaskedBertModel, MaskedBertConfig

sys.path.append(os.path.dirname(__file__))
import test_helper

class TestBertModel(unittest.TestCase):
    def init_data(self, use_cuda) -> None:
        torch.set_grad_enabled(False)
        torch.set_num_threads(4)
        turbo_transformers.set_num_threads(4)
        self.test_device = torch.device('cuda:0') if use_cuda else \
            torch.device('cpu:0')

        self.cfg = BertConfig()
        self.torch_model = BertModel(self.cfg)
        self.torch_model.eval()

        if torch.cuda.is_available():
            self.torch_model.to(self.test_device)

        self.turbo_model = turbo_transformers.SharedBertModel.from_torch(
            self.torch_model, self.test_device, "turbo")


    def load_new_task(self):
        masked_bert_config = MaskedBertConfig()
        masked_bert_model = MaskedBertModel(masked_bert_config)
        if torch.cuda.is_available():
            masked_bert_model.to(self.test_device)

        self.turbo_model.load_new_task_from_torch(masked_bert_model)

    def check_torch_and_turbo(self,
                              use_cuda,
                              use_memory_opt=True):
        
        self.init_data(use_cuda)


        for task_id in range(8):
            self.load_new_task()

        batch_size = 8

        #TODO: random seq_len
        seq_len = 32

        task_ids = torch.LongTensor([0,1,2,3])
        n_samples = torch.LongTensor([2,2,2,2])

        num_iter = 1
        device_name = "GPU" if use_cuda else "CPU"
        input_ids = torch.randint(low=0,
                                  high=self.cfg.vocab_size - 1,
                                  size=(batch_size, seq_len),
                                  dtype=torch.long,
                                  device=self.test_device)


        torch_model = lambda: self.torch_model(input_ids)
        torch_result, torch_qps, torch_time = \
            test_helper.run_model(torch_model, use_cuda, num_iter)
        print(f'BertModel PyTorch({device_name}) QPS {torch_qps}')

        turbo_model = (lambda: self.turbo_model(input_ids, task_ids = task_ids, n_samples = n_samples))

        if use_memory_opt:
            turbo_transformers.bert_opt_mem_allocate_api(
                input_ids.size()[0],  # batch
                input_ids.size()[1],  # seq_len
                self.cfg.num_attention_heads,
                self.cfg.hidden_size,
                self.cfg.num_hidden_layers,
                "GPU" if 'cuda' in input_ids.device.type else "CPU")

        with turbo_transformers.pref_guard("bert_perf") as perf:
            turbo_result, turbo_qps, turbo_time = \
                test_helper.run_model(turbo_model, use_cuda, num_iter)
        print(f'BertModel TurboTransformer({device_name}) QPS {turbo_qps}')

        print(f"batch {batch_size} seq_len {seq_len}")
        print(torch.max(torch_result[0].cpu() - turbo_result[0].cpu()))
        self.assertTrue(
            numpy.allclose(torch_result[0].cpu(),
                           turbo_result[0].cpu(),
                           atol=1e-2,
                           rtol=1e-3))

    def bert_model_test_helper(self, use_memory_opt=False):
        if use_memory_opt:
            turbo_transformers.reset_allocator_schema("model-aware")
        
        print(torch.cuda.is_available())
        print(turbo_transformers.config.is_compiled_with_cuda())

        if torch.cuda.is_available() and \
                turbo_transformers.config.is_compiled_with_cuda():
            self.check_torch_and_turbo(use_cuda=True,
                                        use_memory_opt=use_memory_opt)
        if use_memory_opt:
            turbo_transformers.reset_allocator_schema("naive")

    def test_bert_model(self):
        # self.bert_model_test_helper(True)
        self.bert_model_test_helper(False)


if __name__ == '__main__':
    unittest.main()
