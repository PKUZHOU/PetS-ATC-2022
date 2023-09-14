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

#include "turbo_transformers/layers/bert_output.h"

#include <loguru.hpp>

#include "turbo_transformers/core/memory.h"
#include "turbo_transformers/layers/kernels/common.h"
#include "turbo_transformers/layers/kernels/layer_norm.h"
#include "turbo_transformers/layers/kernels/mat_mul.h"
#include "turbo_transformers/core/tensor.h"

#include "turbo_transformers/layers/kernels/elementwise_add.h"
#include "turbo_transformers/layers/shadow_op.h"

#include "turbo_transformers/core/cuda_utils.h"

#ifdef WITH_PERFTOOLS
#include "turbo_transformers/core/profiler.h"
#endif

namespace turbo_transformers {
namespace layers {

void BertOutput::compute_shadow(const core::Tensor& input_tensor,
                 core::Tensor* output_tensor,
                 const core::Tensor& hidden_states,
                 const core::Tensor* task_ids,
                 const core::Tensor* n_samples,
                 const core::Tensor *minibatch_lens) const{

  core::SHADOW_SYNC_PROLOG();

  #ifdef WITH_PERFTOOLS
  auto& profile_ctx = core::Profiler::GetInstance();
  profile_ctx.start_profile("compute_output_shadow", input_tensor.device_type());
  #endif

  int n_tasks = task_ids -> numel(); // number of total tasks in the batch  
  const int64_t * int_task_ids = task_ids->data<int64_t>(); // convert the task_ids to int list
  const int64_t * int_n_samples = n_samples->data<int64_t>(); //convert the n_samples to int list
  int64_t * int_minibatch_lens = nullptr;
  std::string name = "bert_output";


  bool use_mini_batch = false;
  if(minibatch_lens!=nullptr){
    use_mini_batch = true;
    int_minibatch_lens =const_cast<int64_t*>(minibatch_lens->data<int64_t>());
  }
  int pet_seq_len = input_tensor.shape(1);


  PUSH_RANGE("compute_output_shadow", 0);
  for(int task_idx = 0; task_idx < n_tasks; task_idx ++){ // traverse the tasks in the batch
      int task_id = int_task_ids[task_idx]; // get the task id 
      // define the task-specific tensors
      core::Tensor task_hidden_states(nullptr); // declare the hidden states for current task
      core::Tensor task_output(nullptr);
      core::Tensor task_input(nullptr);
      core::Tensor task_shadow_output(nullptr); // declare the shadow output for current task

      int start = std::accumulate(int_n_samples, int_n_samples + task_idx, 0);
      int end = start + int_n_samples[task_idx]; // current task's minibatch
      

      if (use_mini_batch) { // use minibatch
        pet_seq_len = int_minibatch_lens[task_idx];
      } 

      hidden_states.slice_to(task_hidden_states, start, end, pet_seq_len, 1); // get the hidden_states for current task 
      input_tensor.slice_to(task_input, start, end, pet_seq_len, 1); // get the dense input for current task
      output_tensor->slice_to(task_output, start, end, pet_seq_len, 1); // get the dense output for current task

      if (core::CUDADeviceContext::num_streams > 1) {
        auto cuda_ctx =
            turbo_transformers::core::CUDADeviceContext::GetInstance(task_id);
        task_shadow_output.Reshape<float>({int_n_samples[task_idx], pet_seq_len, 
                                           output_tensor->shape(2)}, output_tensor->device_type(), output_tensor->device_id(),
          cuda_ctx.get(), "BertOutput_Task/Reshape");
      } else {
        task_shadow_output.Reshape<float>({int_n_samples[task_idx], pet_seq_len, 
                                           output_tensor->shape(2)}, output_tensor->device_type(), output_tensor->device_id(),
          "BertOutput_Task/Reshape");
      }
      
      // call the shadow operation
      auto shadow_op = get_shadow_op(pet_layer_manager, task_id);
      shadow_op(pet_layer_manager, task_id, &task_input, &task_output, &task_shadow_output, &task_hidden_states,
        nullptr, nullptr, nullptr,
        false /*Add bias act*/ , true /*layer norm*/, false, /*split_add_transpose*/ name);
  }
  
  core::SHADOW_SYNC_EPILOG();
  
  POP_RANGE;

  #ifdef WITH_PERFTOOLS
  profile_ctx.end_profile("compute_output_shadow", input_tensor.device_type());
  #endif
}



void BertOutput::operator()(const core::Tensor &hidden_states,
                            const core::Tensor &input_tensor,
                            core::Tensor *output_tensor,
                            core::Tensor *task_ids,
                            core::Tensor *n_samples,
                            core::Tensor *minibatch_lens ) const {
#ifdef WITH_PERFTOOLS
  auto &profile_ctx = core::Profiler::GetInstance();
  profile_ctx.start_profile("BertOutput", input_tensor.device_type());
#endif
  TT_ENFORCE_EQ(kernels::common::is_same_device_ctx(input_tensor.device_ctx(),
                                                    hidden_states.device_ctx()),
                true,
                "BertOutput: The input_tensor and hidden_states should have "
                "the same device type and device id.");
  output_tensor->Reshape<float>(
      {hidden_states.shape(0), hidden_states.shape(1), dense_weight_.shape(1)},
      hidden_states.device_type(), hidden_states.device_id(),
      "BERTEmbedding/Reshape");
  // NOTE the out of this bert layer should be the input of the next layer
  //      "BertOutput/Reshape");

  // Compute the shared 
  kernels::MatMul(hidden_states, false, dense_weight_, false, 1.0,
                  output_tensor, 0.0, nullptr, "BertOutput/MatMul");

#if (SHADOW_PERF_DEBUG == 0)
  compute_shadow(input_tensor,
                 output_tensor,
                 hidden_states,
                 task_ids,
                 n_samples,
                 minibatch_lens);
#endif

#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile("BertOutput", input_tensor.device_type());
#endif
}

void BertOutput::EnforceShapeAndType() const {
  if (loguru::current_verbosity_cutoff() >= 3) {
    std::stringstream ss;
    ss << "<<<<<<<< dense_weight_ <<<<<<<<<<";
    dense_weight_.Print<float>(ss);
    ss << "<<<<<<<< dense_bias <<<<<<<<<<";
    dense_bias_.Print<float>(ss);
    ss << "<<<<<<<< layer_norm_weight <<<<<<<<<<";
    layer_norm_weight_.Print<float>(ss);
    ss << "<<<<<<<< layer_norm_bias <<<<<<<<<<";
    layer_norm_bias_.Print<float>(ss);
    LOG_S(3) << ss.str();
  }
}

void BertOutput::load_new_task(
    int & pet_type,
    core::Tensor &task_mask, 
    core::Tensor &task_diff,
    core::Tensor &task_bias,
    core::Tensor &task_layer_norm_weight,
    core::Tensor &task_layer_norm_bias,
    core::Tensor &down_scale_w,
    core::Tensor &down_scale_b,
    core::Tensor &up_scale_w,
    core::Tensor &up_scale_b 
) {
  pet_layer_manager.load_new_task(pet_type, true, &task_mask, &task_diff, &task_bias, &task_layer_norm_weight, 
        &task_layer_norm_bias, &down_scale_w, &down_scale_b, &up_scale_w, &up_scale_b);
}


}  // namespace layers
}  // namespace turbo_transformers
