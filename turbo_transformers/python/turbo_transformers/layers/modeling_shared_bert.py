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

try:
    # `turbo_transformers_cxxd` is the name on debug mode
    import turbo_transformers.turbo_transformers_cxxd as cxx
except ImportError:
    import turbo_transformers.turbo_transformers_cxx as cxx
from typing import Union, Optional, Sequence
import torch
from torch.nn.functional import layer_norm

from .modeling_pets import PET_Types
from .return_type import convert_returns_as_type, ReturnType
from .utils import try_convert, convert2tt_tensor, to_param_dict_convert_tt, to_param_dict, create_empty_if_none, AnyTensor

from transformers.modeling_bert import BertEmbeddings as TorchBertEmbeddings
from transformers.modeling_bert import BertIntermediate as TorchBertIntermediate
from transformers.modeling_bert import BertOutput as TorchBertOutput
from transformers.modeling_bert import BertAttention as TorchBertAttention
from transformers.modeling_bert import BertLayer as TorchBertLayer
from transformers.modeling_bert import BertEncoder as TorchBertEncoder
from transformers.modeling_bert import BertModel as TorchBertModel
from transformers.modeling_bert import BertPooler as TorchBertPooler

from .modeling_pets import PET_Types, PETBertAttention, PETBertIntermediate, PETBertLayer, PETBertOutput, PETBertModel, PETBertEncoder


import enum
import numpy as np

__all__ = [
    'SharedBertModel'
]

def load_maskbert_param(params):
    # maskbert only masks weights
    task_weight_mask = convert2tt_tensor(
            torch.clone(torch.t(params["dense.binary_mask"]).contiguous()))
    return task_weight_mask

def load_diff_param(params, layer_norm = True):
    # diff pruning requires weight difference and bias difference
    # for simplicify, we load the whole bias
    task_diff = convert2tt_tensor(
        torch.clone(torch.t(params["dense.diff"]).contiguous()))
    task_bias = convert2tt_tensor(
        torch.clone(params["dense.bias"]).contiguous())
    if not layer_norm:
        return task_diff, task_bias

    task_layer_norm_weight = convert2tt_tensor(
        torch.clone(params["LayerNorm.weight"]).contiguous())
    task_layer_norm_bias = convert2tt_tensor(
        torch.clone(params["LayerNorm.bias"]).contiguous()) 
    return task_diff, task_bias, task_layer_norm_weight, task_layer_norm_bias

def load_bitfit_param(params, layer_norm = True):
    # biffit requires the bias and layernorm params
    task_bias = convert2tt_tensor(
        torch.clone(params["dense.bias"]).contiguous())
    if not layer_norm:
        return task_bias
    
    task_layer_norm_weight = convert2tt_tensor(
        torch.clone(params["LayerNorm.weight"]).contiguous())
    task_layer_norm_bias = convert2tt_tensor(
        torch.clone(params["LayerNorm.bias"]).contiguous())
    return task_bias, task_layer_norm_weight, task_layer_norm_bias

def load_adapter_param(params, layer_norm = True):
    down_scale_w = convert2tt_tensor(
        torch.clone(params["dense.down_scale_w"]).contiguous())
    down_scale_b = convert2tt_tensor(
        torch.clone(params["dense.down_scale_b"]).contiguous())
    up_scale_w = convert2tt_tensor(
        torch.clone(params["dense.up_scale_w"]).contiguous())
    up_scale_b = convert2tt_tensor(
        torch.clone(params["dense.up_scale_b"]).contiguous())
    if not layer_norm:
        return down_scale_w, down_scale_b,up_scale_w,up_scale_b
    else:
        task_layer_norm_weight = convert2tt_tensor(
            torch.clone(params["LayerNorm.weight"]).contiguous())
        task_layer_norm_bias = convert2tt_tensor(
            torch.clone(params["LayerNorm.bias"]).contiguous())
        return down_scale_w, down_scale_b,up_scale_w,up_scale_b, task_layer_norm_weight, task_layer_norm_bias


class BertEmbeddings(cxx.BERTEmbedding):
    def __call__(self,
                 input_ids: AnyTensor,
                 position_ids: AnyTensor,
                 token_type_ids: AnyTensor,
                 return_type: Optional[ReturnType] = None,
                 output: Optional[cxx.Tensor] = None):
        input_ids = try_convert(input_ids)
        position_ids = try_convert(position_ids)
        token_type_ids = try_convert(token_type_ids)
        output = create_empty_if_none(output)
        super(BertEmbeddings, self).__call__(input_ids, position_ids,
                                             token_type_ids, output)
        return convert_returns_as_type(output, return_type)

    @staticmethod
    def from_torch(bert_embedding: TorchBertEmbeddings) -> 'BertEmbeddings':
        params = to_param_dict_convert_tt(bert_embedding)
        return BertEmbeddings(params['word_embeddings.weight'],
                              params['position_embeddings.weight'],
                              params['token_type_embeddings.weight'],
                              params['LayerNorm.weight'],
                              params['LayerNorm.bias'])

    @staticmethod
    def from_npz(file_name: str, device: Optional[torch.device] = None):
        f = np.load(file_name)
        return BertEmbeddings(
            try_convert(f['embeddings.word_embeddings.weight'], device),
            try_convert(f['embeddings.position_embeddings.weight'], device),
            try_convert(f['embeddings.token_type_embeddings.weight'], device),
            try_convert(f['embeddings.LayerNorm.weight'], device),
            try_convert(f['embeddings.LayerNorm.bias'], device))


class SharedBertIntermediate(cxx.BertIntermediate):
    def __call__(self,
                 input_tensor: AnyTensor,
                 return_type: Optional[ReturnType] = None,
                 output: Optional[cxx.Tensor] = None,
                 
                 task_ids: AnyTensor = None,
                 n_samples: AnyTensor = None,
                 minibatch_lens: AnyTensor = None,
                 ):
        input_tensor = try_convert(input_tensor)
        output = create_empty_if_none(output)
        super(SharedBertIntermediate, self).__call__(input_tensor, output, task_ids, n_samples, minibatch_lens)
        return convert_returns_as_type(output, return_type)

    @staticmethod
    def from_torch(intermediate: TorchBertIntermediate):
        intermediate_params = to_param_dict(intermediate)
        weight = torch.clone(
            torch.t(intermediate_params["dense.weight"]).contiguous())
        return SharedBertIntermediate(
            convert2tt_tensor(weight),
            convert2tt_tensor(intermediate_params['dense.bias']))

    @staticmethod
    def from_npz(file_name: str,
                 layer_num: int,
                 device: Optional[torch.device] = None):
        f = np.load(file_name)
        return SharedBertIntermediate(
            try_convert(
                f[f'encoder.layer.{layer_num}.intermediate.dense.weight'],
                device),
            try_convert(
                f[f'encoder.layer.{layer_num}.intermediate.dense.bias'],
                device))

    def load_new_task_from_torch(self, intermediate: PETBertIntermediate):
        intermediate_params = to_param_dict(intermediate)
        
        pet_type = intermediate.pet_type
        
        task_weight_mask = create_empty_if_none(None)
        task_diff = create_empty_if_none(None)
        task_bias = create_empty_if_none(None)

        if pet_type == PET_Types.maskbert:
            # maskbert only masks weights
            task_weight_mask = load_maskbert_param(intermediate_params)

        elif pet_type == PET_Types.diff_pruning:
            task_diff, task_bias = load_diff_param(intermediate_params, layer_norm = False)

        elif pet_type == PET_Types.bitfit:
            task_bias = load_bitfit_param(intermediate_params, layer_norm = False)
        
        elif pet_type == PET_Types.adapters:
            pass

        # call cxx module
        self.load_new_task(int(pet_type), task_weight_mask, task_diff,  task_bias)




class SharedBertOutput(cxx.BertOutput):
    def __call__(self,
                 intermediate_output: AnyTensor,
                 attention_output: AnyTensor,
                 return_type: Optional[ReturnType] = None,
                 output: Optional[cxx.Tensor] = None,
                 
                 task_ids: AnyTensor = None,
                 n_samples: AnyTensor = None,
                 minibatch_lens: AnyTensor = None,
                 ):
        intermediate_output = try_convert(intermediate_output)
        attention_output = try_convert(attention_output)
        output = create_empty_if_none(output)

        super(SharedBertOutput, self).__call__(intermediate_output, attention_output,
                                         output, task_ids, n_samples, minibatch_lens)
        return convert_returns_as_type(output, return_type)


    @staticmethod
    def from_torch(output: TorchBertOutput):
        params = to_param_dict(output)
        weight = convert2tt_tensor(
            torch.clone(torch.t(params["dense.weight"]).contiguous()))
        return SharedBertOutput(weight, convert2tt_tensor(params["dense.bias"]),
                          convert2tt_tensor(params["LayerNorm.weight"]),
                          convert2tt_tensor(params["LayerNorm.bias"]))

    @staticmethod
    def from_npz(file_name: str,
                 layer_num: int,
                 device: Optional[torch.device] = None):
        f = np.load(file_name)
        return SharedBertOutput(
            try_convert(f[f'encoder.layer.{layer_num}.output.dense.weight'],
                        device),
            try_convert(f[f'encoder.layer.{layer_num}.output.dense.bias'],
                        device),
            try_convert(
                f[f'encoder.layer.{layer_num}.output.LayerNorm.weight'],
                device),
            try_convert(f[f'encoder.layer.{layer_num}.output.LayerNorm.bias'],
                        device))

    def load_new_task_from_torch(self, output: PETBertOutput):
        params = to_param_dict(output)
        pet_type = output.pet_type
        
        task_weight_mask = create_empty_if_none(None)
        task_diff = create_empty_if_none(None)
        task_bias = create_empty_if_none(None)
        task_layer_norm_weight = create_empty_if_none(None)
        task_layer_norm_bias = create_empty_if_none(None)
        #for adapters
        down_scale_w = create_empty_if_none(None)
        down_scale_b = create_empty_if_none(None)
        up_scale_w = create_empty_if_none(None)
        up_scale_b = create_empty_if_none(None)

        if pet_type == PET_Types.maskbert:
            task_weight_mask = load_maskbert_param(params)
        elif pet_type == PET_Types.diff_pruning:
            task_diff, task_bias, task_layer_norm_weight, task_layer_norm_bias = load_diff_param(params)
        elif pet_type == PET_Types.bitfit:
            task_bias, task_layer_norm_weight, task_layer_norm_bias = load_bitfit_param(params)
        elif pet_type == PET_Types.adapters:
            down_scale_w, down_scale_b, up_scale_w, up_scale_b, task_layer_norm_weight, task_layer_norm_bias = load_adapter_param(params)

        # call cxx module
        self.load_new_task(int(pet_type), task_weight_mask, task_diff, task_bias, task_layer_norm_weight, task_layer_norm_bias, down_scale_w, down_scale_b, up_scale_w, up_scale_b)

class SharedBertAttention(cxx.BertAttention):
    def __call__(self,
                 input_tensor: AnyTensor,
                 attention_mask: Optional[AnyTensor] = None,
                 head_mask: Optional[AnyTensor] = None,
                 output_attentions: Optional[bool] = False,
                 return_type: Optional[ReturnType] = None,
                 is_trans_weight: Optional[cxx.Tensor] = False,
                 
                 #for multi_task
                 task_ids: AnyTensor = None,
                 n_samples: AnyTensor = None,
                 minibatch_lens: AnyTensor = None
                 ):
        """
        implement BertSelfAttention in
        https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_bert.py#L183
        self.output_attentions always true
        return (context_layer, attention_probs)
        """
        assert (head_mask is None)
        input_tensor = try_convert(input_tensor)
        attention_mask = try_convert(create_empty_if_none(attention_mask))
        context_layer = cxx.Tensor.create_empty()
        attn_probs = cxx.Tensor.create_empty()
        super(SharedBertAttention,
              self).__call__(input_tensor, attention_mask, context_layer,
                             attn_probs, is_trans_weight, task_ids, n_samples, minibatch_lens)
        outputs = (convert_returns_as_type(context_layer, return_type),
                   convert_returns_as_type(attn_probs, ReturnType.TORCH)
                   ) if output_attentions else (convert_returns_as_type(
                       context_layer, return_type), )
        return outputs

    @staticmethod
    def from_torch(attention: TorchBertAttention):
        params = {k: v for k, v in attention.named_parameters()}
        with torch.no_grad():
            # merge self.query.weight, self.query.weight and self.query.weight together as qkv.weight
            qkv_weight = torch.clone(
                torch.t(
                    torch.cat((params['self.query.weight'],
                               params['self.key.weight'],
                               params['self.value.weight']),
                              0).contiguous()).contiguous())
            qkv_bias = torch.cat(
                (params['self.query.bias'], params['self.key.bias'],
                 params['self.value.bias']), 0).contiguous()

            output_weight = torch.clone(
                torch.t(params['output.dense.weight']).contiguous())
            att = SharedBertAttention(
                convert2tt_tensor(qkv_weight), convert2tt_tensor(qkv_bias),
                convert2tt_tensor(output_weight),
                convert2tt_tensor(params['output.dense.bias']),
                convert2tt_tensor(params['output.LayerNorm.weight']),
                convert2tt_tensor(params['output.LayerNorm.bias']),
                attention.self.num_attention_heads)

            return att

    @staticmethod
    def from_npz(file_name: str,
                 layer_num: int,
                 num_attention_heads: int,
                 device: Optional[torch.device] = None):
        f = np.load(file_name)
        return SharedBertAttention(
            try_convert(f[f'encoder.layer.{layer_num}.attention.qkv.weight'],
                        device),
            try_convert(f[f'encoder.layer.{layer_num}.attention.qkv.bias'],
                        device),
            try_convert(
                f[f'encoder.layer.{layer_num}.attention.output.dense.weight'],
                device),
            try_convert(
                f[f'encoder.layer.{layer_num}.attention.output.dense.bias'],
                device),
            try_convert(
                f[f'encoder.layer.{layer_num}.attention.output.LayerNorm.weight'],
                device),
            try_convert(
                f[f'encoder.layer.{layer_num}.attention.output.LayerNorm.bias'],
                device), num_attention_heads)

    def load_new_task_from_torch(self, attention: PETBertAttention):
        params = to_param_dict(attention)
        pet_type = attention.pet_type

        with torch.no_grad():
            # merge self.query.weight, self.query.weight and self.query.weight together as qkv.weight
            qkv_weight_mask = create_empty_if_none(None)
            qkv_weight_diff = create_empty_if_none(None)
            qkv_bias = create_empty_if_none(None)
            output_bias = create_empty_if_none(None)
            output_weight_mask = create_empty_if_none(None)
            output_weight_diff = create_empty_if_none(None)
            output_layerNorm_weight = create_empty_if_none(None)
            output_layerNorm_bias = create_empty_if_none(None)
            down_scale_w = create_empty_if_none(None)
            down_scale_b = create_empty_if_none(None)
            up_scale_w = create_empty_if_none(None)
            up_scale_b = create_empty_if_none(None)

            if pet_type == PET_Types.maskbert:
                qkv_weight_mask = convert2tt_tensor(torch.clone(
                    torch.t(
                        torch.cat((params['self.query.binary_mask'],
                                params['self.key.binary_mask'],
                                params['self.value.binary_mask']),
                                0).contiguous()).contiguous()))
                output_weight_mask = convert2tt_tensor(torch.clone(
                    torch.t(params['output.dense.binary_mask']).contiguous()))
            
            elif pet_type == PET_Types.diff_pruning:
                qkv_weight_diff = convert2tt_tensor(torch.clone(
                    torch.t(
                        torch.cat((params['self.query.diff'],
                                params['self.key.diff'],
                                params['self.value.diff']),
                                0).contiguous()).contiguous()))    
                qkv_bias = convert2tt_tensor(torch.cat(
                    (params['self.query.bias'], params['self.key.bias'],
                    params['self.value.bias']), 0).contiguous())
                output_weight_diff = convert2tt_tensor(torch.clone(
                    torch.t(params['output.dense.diff']).contiguous()))
                output_bias = convert2tt_tensor(params['output.dense.bias'])
                output_layerNorm_weight =  convert2tt_tensor(params['output.LayerNorm.weight'])
                output_layerNorm_bias = convert2tt_tensor(params['output.LayerNorm.bias'])

            elif pet_type == PET_Types.bitfit:
                qkv_bias = convert2tt_tensor(torch.cat(
                    (params['self.query.bias'], params['self.key.bias'],
                    params['self.value.bias']), 0).contiguous())
                output_bias = convert2tt_tensor(params['output.dense.bias'])
                output_layerNorm_weight =  convert2tt_tensor(params['output.LayerNorm.weight'])
                output_layerNorm_bias = convert2tt_tensor(params['output.LayerNorm.bias'])

            elif pet_type == PET_Types.adapters:
                output_layerNorm_weight =  convert2tt_tensor(params['output.LayerNorm.weight'])
                output_layerNorm_bias = convert2tt_tensor(params['output.LayerNorm.bias'])
                down_scale_w = convert2tt_tensor(params['output.dense.down_scale_w'])
                down_scale_b = convert2tt_tensor(params['output.dense.down_scale_b'])
                up_scale_w = convert2tt_tensor(params['output.dense.up_scale_w'])
                up_scale_b = convert2tt_tensor(params['output.dense.up_scale_b'])

            # TODO: real mask
            self.load_new_task(int(pet_type), qkv_weight_mask, qkv_weight_diff, qkv_bias,
                output_weight_mask,
                output_weight_diff,
                output_bias, 
                output_layerNorm_weight,
                output_layerNorm_bias,
                down_scale_w,
                down_scale_b,
                up_scale_w,
                up_scale_b)
        
class SharedBertLayer:
    def __init__(self, attention: SharedBertAttention,
                 intermediate: SharedBertIntermediate, output: SharedBertOutput):
        self.attention = attention
        self.intermediate = intermediate
        self.output = output

    def __call__(self,
                 hidden_states: AnyTensor,
                 attention_mask: Optional[AnyTensor] = None,
                 head_mask: Optional[AnyTensor] = None,
                 output_attentions=False,
                 return_type: Optional[ReturnType] = None,
                 
                 #for multi_task
                 task_ids: AnyTensor = None,
                 n_samples: AnyTensor = None,
                 minibatch_lens: AnyTensor = None,
                 ):

        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            return_type=ReturnType.turbo_transformers,
            task_ids = task_ids,
            n_samples = n_samples,
            minibatch_lens= minibatch_lens)
            
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        intermediate_output = self.intermediate(
            attention_output, return_type=ReturnType.turbo_transformers,
            task_ids = task_ids,
            n_samples = n_samples,
            minibatch_lens = minibatch_lens
            )
        
        layer_output = self.output(intermediate_output,
                                   attention_output,
                                   return_type=return_type,
                                   task_ids = task_ids,
                                   n_samples = n_samples,
                                   minibatch_lens = minibatch_lens)
        
        outputs = (layer_output, ) + outputs
        return outputs

    @staticmethod
    def from_torch(layer: TorchBertLayer):
        return SharedBertLayer(SharedBertAttention.from_torch(layer.attention),
                         SharedBertIntermediate.from_torch(layer.intermediate),
                         SharedBertOutput.from_torch(layer.output))

    @staticmethod
    def from_npz(file_name: str,
                 layer_num: int,
                 num_attention_heads: int,
                 device: Optional[torch.device] = None):
        f = np.load(file_name)
        return SharedBertLayer(
            SharedBertAttention.from_npz(file_name, layer_num, num_attention_heads,
                                   device),
            SharedBertIntermediate.from_npz(file_name, layer_num, device),
            SharedBertOutput.from_npz(file_name, layer_num, device))

    def load_new_task_from_torch(self, layer: PETBertLayer):
        self.intermediate.load_new_task_from_torch(layer.intermediate)
        self.attention.load_new_task_from_torch(layer.attention)
        self.output.load_new_task_from_torch(layer.output)


class SharedBertEncoder:
    def __init__(self, layer: Sequence[SharedBertLayer]):
        self.layer = layer

    def __call__(self,
                 hidden_states: AnyTensor,
                 attention_mask: Optional[AnyTensor] = None,
                 head_mask: Optional[AnyTensor] = None,
                 output_attentions: Optional[bool] = False,
                 output_hidden_states: Optional[bool] = False,
                 return_type: Optional[ReturnType] = None,
                 
                 #for multi_task
                 task_ids: AnyTensor = None,
                 n_samples: AnyTensor = None,
                 minibatch_lens: AnyTensor = None
                 ):
        
        all_hidden_states = ()
        all_attentions = ()
        hidden_states = try_convert(hidden_states)
        for l in self.layer:
            layer_outputs = l(hidden_states=hidden_states,
                              attention_mask=attention_mask,
                              output_attentions=output_attentions,
                              return_type=ReturnType.turbo_transformers,
                              task_ids = task_ids,
                              n_samples = n_samples,
                              minibatch_lens = minibatch_lens)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (
                    convert_returns_as_type(hidden_states, ReturnType.TORCH), )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1], )

        outputs = (convert_returns_as_type(hidden_states, return_type), )
        # Add last layer
        if output_hidden_states:
            # TODO(jiaruifang)two return value use the same memory space, that is not supported in dlpack.
            # So we do not append the last hidden_state at the buttom of all_hidden_states,
            # User should use outputs[0] if necessary
            # all_hidden_states = all_hidden_states + (convert_returns_as_type(hidden_states, ReturnType.TORCH),)
            pass

        if output_hidden_states:
            outputs = outputs + (all_hidden_states, )
        if output_attentions:
            outputs = outputs + (all_attentions, )

        return outputs

    @staticmethod
    def from_torch(encoder: TorchBertEncoder):
        layer = [
            SharedBertLayer.from_torch(bert_layer) for bert_layer in encoder.layer
        ]
        return SharedBertEncoder(layer)

    @staticmethod
    def from_npz(file_name: str,
                 num_hidden_layers: int,
                 num_attention_heads: int,
                 device: Optional[torch.device] = None):
        layer = []
        for i in range(num_hidden_layers):
            layer.append(
                SharedBertLayer.from_npz(file_name, i, num_attention_heads, device))
        return SharedBertEncoder(layer)

    def load_new_task_from_torch(self, encoder : PETBertEncoder):
        for layer_idx, bert_layer in enumerate(encoder.layer):
            self.layer[layer_idx].load_new_task_from_torch(bert_layer)

class SequencePool(cxx.SequencePool):
    def __call__(self,
                 input_tensor: AnyTensor,
                 return_type: Optional[ReturnType] = None,
                 output_tensor: Optional[cxx.Tensor] = None):
        input_tensor = try_convert(input_tensor)
        output_tensor = create_empty_if_none(output_tensor)
        super(SequencePool, self).__call__(input_tensor, output_tensor)
        return convert_returns_as_type(output_tensor, return_type)


class PoolingType(enum.Enum):
    FIRST = "First"
    LAST = "Last"
    MEAN = "Mean"
    MAX = "Max"


PoolingMap = {
    PoolingType.FIRST: "First",
    PoolingType.LAST: "Last",
    PoolingType.MEAN: "Mean",
    PoolingType.MAX: "Max"
}


class BertPooler(cxx.BertPooler):
    def __call__(self,
                 input_tensor: AnyTensor,
                 return_type: Optional[ReturnType] = None,
                 output: Optional[cxx.Tensor] = None):
        input_tensor = try_convert(input_tensor)
        output = create_empty_if_none(output)
        super(BertPooler, self).__call__(input_tensor, output)
        return convert_returns_as_type(output, return_type)

    @staticmethod
    def from_torch(pooler: TorchBertPooler):
        pooler_params = to_param_dict(pooler)
        weight = torch.clone(
            torch.t(pooler_params['dense.weight']).contiguous())
        return BertPooler(convert2tt_tensor(weight),
                          convert2tt_tensor(pooler_params['dense.bias']))

    @staticmethod
    def from_npz(file_name: str, device: Optional[torch.device] = None):
        f = np.load(file_name)
        return BertPooler(try_convert(f['pooler.dense.weight'], device),
                          try_convert(f['pooler.dense.bias'], device))


class SharedBertModelNoPooler:
    def __init__(self, embeddings: BertEmbeddings, encoder: SharedBertEncoder):
        self.embeddings = embeddings
        self.encoder = encoder
        self.prepare = cxx.PrepareBertMasks()

    def __call__(
            self,
            inputs: AnyTensor,
            attention_masks: Optional[AnyTensor] = None,
            token_type_ids: Optional[AnyTensor] = None,
            position_ids: Optional[AnyTensor] = None,
            head_mask: Optional[AnyTensor] = None,
            inputs_embeds: Optional[AnyTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            pooling_type: PoolingType = PoolingType.
            FIRST,  #the following parameters are exclusive for turbo
            return_type: Optional[ReturnType] = None,
            
            #for multi-task
            task_ids : AnyTensor = None,
            n_samples : AnyTensor = None,
            minibatch_lens: AnyTensor = None
            ):
        attention_masks = try_convert(create_empty_if_none(attention_masks))
        token_type_ids = try_convert(create_empty_if_none(token_type_ids))
        position_ids = try_convert(create_empty_if_none(position_ids))
        inputs = try_convert(inputs)
        extended_attention_masks = cxx.Tensor.create_empty()

        self.prepare(inputs, attention_masks, token_type_ids, position_ids,
                     extended_attention_masks)

        hidden_cache = self.embeddings(
            inputs,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            return_type=ReturnType.turbo_transformers)

        encoder_outputs = self.encoder(
            hidden_states=hidden_cache,
            attention_mask=extended_attention_masks,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_type=return_type,
            task_ids = task_ids,
            n_samples = n_samples,
            minibatch_lens = minibatch_lens
            )
        return encoder_outputs

    @staticmethod
    def from_torch(model: TorchBertModel,
                   device: Optional[torch.device] = None):
        if device is not None and 'cuda' in device.type and torch.cuda.is_available(
        ):
            model.to(device)
        embeddings = BertEmbeddings.from_torch(model.embeddings)
        encoder = SharedBertEncoder.from_torch(model.encoder)
        return SharedBertModelNoPooler(embeddings, encoder)

    @staticmethod
    def from_pretrained(model_id_or_path: str,
                        device: Optional[torch.device] = None):
        torch_model = TorchBertModel.from_pretrained(model_id_or_path)
        model = SharedBertModelNoPooler.from_torch(torch_model, device)
        model.config = torch_model.config
        model._torch_model = torch_model  # prevent destroy torch model.
        return model

    @staticmethod
    def from_npz(file_name: str, config,
                 device: Optional[torch.device] = None):
        embeddings = BertEmbeddings.from_npz(file_name, device)
        encoder = SharedBertEncoder.from_npz(file_name, config.num_hidden_layers,
                                       config.num_attention_heads, device)
        return SharedBertModelNoPooler(embeddings, encoder)

    def load_new_task_from_torch(self, model: PETBertModel,
                   device: Optional[torch.device] = None):
        if device is not None and 'cuda' in device.type and torch.cuda.is_available(
        ):
            model.to(device)
        
        self.encoder.load_new_task_from_torch(model.encoder)

class SharedBertModel:
    # @params:
    # pooler is used for turbo backend only
    # config is used for memory optizations
    def __init__(self, model, pooler=None, backend="onnxrt", config=None):
        # TODO type of bertmodel_nopooler is (onnx and torch)
        self.backend = backend
        # only support turbo
        assert backend == "turbo"
        self.config = config
        self.shared_bertmodel_nopooler = model
        self.pooler = pooler
        self.backend = "turbo"

    def __call__(self,
                 inputs: AnyTensor,
                 attention_masks: Optional[AnyTensor] = None,
                 token_type_ids: Optional[AnyTensor] = None,
                 position_ids: Optional[AnyTensor] = None,
                 head_mask: Optional[AnyTensor] = None,
                 inputs_embeds: Optional[AnyTensor] = None,
                 output_attentions: Optional[bool] = None,
                 output_hidden_states: Optional[bool] = None,
                 pooling_type: PoolingType = PoolingType.FIRST,
                 pooler_output: Optional[AnyTensor] = None,
                 return_type: Optional[ReturnType] = None,

                 # for specific tasks
                 task_ids: Optional[AnyTensor] = None,
                 n_samples: Optional[AnyTensor] = None,
                 minibatch_lens: Optional[AnyTensor] = None
                 ):
        
        # convert to tt tensor
        task_ids = try_convert(task_ids)
        n_samples = try_convert(n_samples)
        minibatch_lens = try_convert(minibatch_lens)

        encoder_outputs = self.shared_bertmodel_nopooler(
            inputs,
            attention_masks,
            token_type_ids,
            position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            pooling_type=pooling_type,
            return_type=ReturnType.turbo_transformers,

            #for specific tasks
            task_ids = task_ids,
            n_samples = n_samples,
            minibatch_lens = minibatch_lens
            )

        sequence_output = encoder_outputs[0]
        self.seq_pool = SequencePool(PoolingMap[pooling_type])
        sequence_pool_output = self.seq_pool(
            input_tensor=sequence_output,
            return_type=ReturnType.turbo_transformers)
        pooler_output = self.pooler(sequence_pool_output, return_type,
                                    pooler_output)
        return (
            convert_returns_as_type(sequence_output, return_type),
            pooler_output,
        ) + encoder_outputs[1:]
    

    @staticmethod
    def from_torch(model: TorchBertModel,
                    device: Optional[torch.device] = None,
                    backend: Optional[str] = None,
                    use_memory_opt=False):
        """
        Args:
            model : a PyTorch Bert Model
            device : cpu or GPU
            backend : a string to indicates kernel provides
            Four options. [onnxrt-cpu, onnxrt-gpu, turbo-cpu, turbo-gpu]
            use_memory_opt [bool] whether or not use memory opt for variable length inputs.
        """
        embeddings = BertEmbeddings.from_torch(model.embeddings)
        encoder = SharedBertEncoder.from_torch(model.encoder)
        bertmodel_nopooler = SharedBertModelNoPooler(embeddings, encoder)
        pooler = BertPooler.from_torch(model.pooler)
        return SharedBertModel(bertmodel_nopooler, pooler, "turbo", model.config)
        
    @staticmethod
    def from_pretrained(model_id_or_path: str,
                        device: Optional[torch.device] = None,
                        backend: Optional[str] = None):
        torch_model = TorchBertModel.from_pretrained(model_id_or_path)
        model = SharedBertModel.from_torch(torch_model, device, backend,
                                     torch_model.config)
        model.config = torch_model.config
        model._torch_model = torch_model  # prevent destroy torch model.
        return model

    @staticmethod
    def from_npz(file_name: str, config,
                 device: Optional[torch.device] = None):
        model = SharedBertModelNoPooler.from_npz(file_name, config, device)
        pooler = BertPooler.from_npz(file_name, device)
        return SharedBertModel(model, pooler, backend="turbo")
    
    def load_new_task_from_torch(self, model: PETBertModel):
        # each task has the same W 
        # each task has different layerNorm, scale, bias and classification layer
        # each task has task-specific weight mask
        self.shared_bertmodel_nopooler.load_new_task_from_torch(model)