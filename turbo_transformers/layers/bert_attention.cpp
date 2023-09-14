// Copyright (C) 2020 THL A29 Limited, a Tencent company.
// All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may
// not use this file except in compliance with the License. You may
// obtain a copy of the License at
// https://opensource.org/licenses/BSD-3-Clause
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" basis,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.
// See the AUTHORS file for names of contributors.

#include "turbo_transformers/layers/bert_attention.h"

#include <unordered_map>

#include "loguru.hpp"
#include "turbo_transformers/core/memory.h"
#include "turbo_transformers/layers/kernels/common.h"
#include "turbo_transformers/layers/kernels/layer_norm.h"
#include "turbo_transformers/layers/kernels/mat_mul.h"
#include "turbo_transformers/layers/kernels/softmax.h"
#include "turbo_transformers/layers/kernels/transpose.h"

namespace turbo_transformers {
namespace layers {

void BertAttention::operator()(const core::Tensor& input_tensor,
                               const core::Tensor& attention_mask,
                               core::Tensor* output, core::Tensor* attn,
                               bool is_trans_weight,
                               
                               core::Tensor *task_ids,
                               core::Tensor *n_samples,
                               core::Tensor *minibatch_lens
                               ) const {
  std::unordered_map<std::string, core::Tensor*> dummy{};
  core::Tensor* attn_ptr;
  if (attn == nullptr) {
    attn_ptr = new core::Tensor(nullptr);
  } else {
    attn_ptr = attn;
  }
  MultiHeadedAttention::operator()(
      input_tensor, input_tensor, input_tensor, attention_mask, "self", output,
      attn_ptr, dummy, false /* pre_layernorm */, true /* post_layernorm */,
      false /* post_add_input */, is_trans_weight /* is_trans_weight */ , task_ids, n_samples, minibatch_lens);
  if (attn == nullptr) {
    delete attn_ptr;
  }
}

void BertAttention::EnforceShapeAndType() const {
  MultiHeadedAttention::EnforceShapeAndType();
}

void BertAttention::load_new_task(  
    int & pet_type,
    core::Tensor &task_qkv_weight_mask, 
    core::Tensor &task_qkv_weight_diff,
    core::Tensor &task_qkv_bias,
    core::Tensor &task_output_weight_mask,
    core::Tensor &task_output_weight_diff,
    core::Tensor &task_output_bias,
    core::Tensor &task_layer_norm_weight,
    core::Tensor &task_layer_norm_bias,
    core::Tensor &down_scale_w,
    core::Tensor &down_scale_b,
    core::Tensor &up_scale_w,
    core::Tensor &up_scale_b 
) {
    int qkv_type = (pet_type == ADAPTERS) ? STANDARD : pet_type;
    qkv_manager.load_new_task(qkv_type, false, &task_qkv_weight_mask, &task_qkv_weight_diff, &task_qkv_bias, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    output_manager.load_new_task(pet_type, true, &task_output_weight_mask, &task_output_weight_diff, &task_output_bias, &task_layer_norm_weight, &task_layer_norm_bias, 
      &down_scale_w, &down_scale_b, &up_scale_w, &up_scale_b);
}

BertAttention::~BertAttention() {
}

}  // namespace layers
}  // namespace turbo_transformers
