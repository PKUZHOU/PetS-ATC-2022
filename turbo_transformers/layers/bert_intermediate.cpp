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

#include "turbo_transformers/layers/bert_intermediate.h"

#include <loguru.hpp>

#include "turbo_transformers/core/blas.h"
#include "turbo_transformers/core/memory.h"
#include "turbo_transformers/layers/kernels/activation.h"
#include "turbo_transformers/layers/kernels/layer_norm.h"
#include "turbo_transformers/layers/kernels/mat_mul.h"
#include "turbo_transformers/layers/kernels/softmax.h"
#include "turbo_transformers/layers/kernels/transpose.h"
#include "turbo_transformers/layers/kernels/elementwise_add.h"
#include "turbo_transformers/layers/shadow_op.h"

#include "turbo_transformers/core/cuda_utils.h"

#ifdef WITH_PERFTOOLS
#include "turbo_transformers/core/profiler.h"
#endif

namespace turbo_transformers {
namespace layers {


void BertIntermediate::compute_shadow(const core::Tensor& input_tensor,
                                      core::Tensor* output_tensor,
                                      const core::Tensor* task_ids,
                                      const core::Tensor* n_samples,
                                      const core::Tensor* minibatch_lens) const{

  core::SHADOW_SYNC_PROLOG();

  #ifdef WITH_PERFTOOLS
  auto& profile_ctx = core::Profiler::GetInstance();
  profile_ctx.start_profile("compute_intermediate_shadow", input_tensor.device_type());
  #endif

  int n_tasks = task_ids -> numel(); // number of total tasks in the batch  
  const int64_t * int_task_ids = task_ids->data<int64_t>(); // convert the task_ids to int list
  const int64_t * int_n_samples = n_samples->data<int64_t>(); //convert the n_samples to int list
  int64_t * int_minibatch_lens = nullptr;
  std::string name = "bert_intermediate";

  bool use_mini_batch = false;
  if(minibatch_lens!=nullptr){
    use_mini_batch = true;
    int_minibatch_lens =const_cast<int64_t*>(minibatch_lens->data<int64_t>());
  }
  int pet_seq_len = input_tensor.shape(1);


  PUSH_RANGE("compute_intermediate_shadow", 0);
  for(int task_idx = 0; task_idx < n_tasks; task_idx ++){ // traverse the tasks in the batch
      int task_id = int_task_ids[task_idx]; // get the task id 
      auto shadow_op = get_shadow_op(pet_layer_manager, task_id);
      
      int start = std::accumulate(int_n_samples, int_n_samples + task_idx, 0);
      int end = start + int_n_samples[task_idx]; // current task's minibatch
      
      // task specific tensors
      core::Tensor task_output(nullptr);
      core::Tensor task_input(nullptr);
      core::Tensor task_shadow_output(nullptr); // declare the shadow output for current task

      if (use_mini_batch) { // use minibatch
        pet_seq_len = int_minibatch_lens[task_idx];
      } 

      input_tensor.slice_to(task_input, start, end, pet_seq_len, 1); // get the dense input for current task
      output_tensor->slice_to(task_output, start, end, pet_seq_len, 1); // get the dense output for current task

      if (core::CUDADeviceContext::num_streams > 1) {
        auto cuda_ctx =
            turbo_transformers::core::CUDADeviceContext::GetInstance(task_id);
        task_shadow_output.Reshape<float>({int_n_samples[task_idx], pet_seq_len, 
                                           output_tensor->shape(2)}, output_tensor->device_type(),
          output_tensor->device_id(),
          cuda_ctx.get(), "BertIntermediate_Task/Reshape");
      } else {
        task_shadow_output.Reshape<float>({int_n_samples[task_idx], pet_seq_len, 
                                           output_tensor->shape(2)}, output_tensor->device_type(),
          output_tensor->device_id(),
          "BertIntermediate_Task/Reshape");
      }

      // call the shadow operator
      shadow_op(pet_layer_manager, task_id, &task_input, &task_output, &task_shadow_output, nullptr, nullptr, nullptr, nullptr,
         true /*Add bias act*/ , false /*no layer norm*/,false /*split add transpose*/,  name);
  }

  core::SHADOW_SYNC_EPILOG();
  
  POP_RANGE;

  #ifdef WITH_PERFTOOLS
  profile_ctx.end_profile("compute_intermediate_shadow", input_tensor.device_type());
  #endif
}


void BertIntermediate::operator()(const core::Tensor& input_tensor,
                                  core::Tensor* output_tensor,
                                  core::Tensor* task_ids,
                                  core::Tensor* n_samples,
                                  core::Tensor* minibatch_lens ) const {
#ifdef WITH_PERFTOOLS
  auto& profile_ctx = core::Profiler::GetInstance();
  profile_ctx.start_profile("BertIntermediate", input_tensor.device_type());
#endif
  output_tensor->Reshape<float>(
      {input_tensor.shape(0), input_tensor.shape(1), dense_weight_.shape(1)},
      input_tensor.device_type(), input_tensor.device_id(),
      "BertIntermediate/Reshape");


  // Shared dense kernel
  kernels::MatMul(input_tensor, false, dense_weight_, false, 1.0, output_tensor,
                  0.0, "BertIntermediate/MatMul");

  // Task-specific operations
  #if (SHADOW_PERF_DEBUG == 0)
  compute_shadow(input_tensor, output_tensor, task_ids, n_samples, minibatch_lens);
  #endif


#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile("BertIntermediate", input_tensor.device_type());
#endif
}

void BertIntermediate::EnforceShapeAndType() const {
  TT_ENFORCE_EQ(dense_weight_.n_dim(), 2, "dense weight must be matrix");
  TT_ENFORCE_EQ(dense_bias_.n_dim(), 1, "dense bias must be vector");
  TT_ENFORCE_EQ(dense_weight_.shape(1), dense_bias_.shape(0),
                "weight and bias shape mismatch %d, %d", dense_weight_.shape(1),
                dense_bias_.shape(0));

  if (loguru::current_verbosity_cutoff() >= 3) {
    std::ostringstream os;
    os << ">>>>>>>>>>>> query_weight <<<<<<<<<<<<" << std::endl;
    dense_weight_.Print<float>(os);
    os << ">>>>>>>>>>>> query_bias <<<<<<<<<<<<" << std::endl;
    dense_bias_.Print<float>(os);
    LOG_S(3) << os.str();
  }
}

void BertIntermediate::load_new_task(
    int & pet_type,
    core::Tensor &task_mask,
    core::Tensor &task_diff,
    core::Tensor &task_bias) {
    
    int type = (pet_type == ADAPTERS)? STANDARD : pet_type;
    pet_layer_manager.load_new_task(type, false, &task_mask, &task_diff, &task_bias, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
}

}  // namespace layers
}  // namespace turbo_transformers
