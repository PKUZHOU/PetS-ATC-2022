#pragma once
#include "turbo_transformers/core/tensor.h"
#include "turbo_transformers/core/pet_manager.h"
#include "turbo_transformers/core/cuda_device_context.h"
#include "turbo_transformers/layers/kernels/common.h"
#include "turbo_transformers/layers/kernels/layer_norm.h"
#include "turbo_transformers/layers/kernels/mat_mul.h"
#include "turbo_transformers/layers/kernels/elementwise_add.h"
#include "turbo_transformers/layers/kernels/activation.h"
#include "turbo_transformers/layers/kernels/transpose.h"

namespace turbo_transformers {
namespace layers {

typedef void(* shadow_op) (const core::PETLayerManager&, const int, 
                           core::Tensor*, core::Tensor*, core::Tensor*, 
                           core::Tensor*, core::Tensor*, core::Tensor*, core::Tensor*,    
                           bool, bool, bool, std::string&);

void compute_mask_shadow(
            const core::PETLayerManager& pet_layer_manager,
            const int task_id,
            core::Tensor* task_input,
            core::Tensor* task_output, 
            core::Tensor* task_shadow_output,
            core::Tensor* task_hidden_states,
            core::Tensor* task_q_out,
            core::Tensor* task_k_out,
            core::Tensor* task_v_out,

            bool add_bias_act,  bool add_input_bias_layernorm, bool split_add_transpose, std::string&name);

void compute_diff_shadow(
            const core::PETLayerManager& pet_layer_manager,
            const int task_id,
            core::Tensor* task_input,
            core::Tensor* task_output, 
            core::Tensor* task_shadow_output,
            core::Tensor* task_hidden_states,
            core::Tensor* task_q_out,
            core::Tensor* task_k_out,
            core::Tensor* task_v_out,

            bool add_bias_act,  bool add_input_bias_layernorm, bool split_add_transpose, std::string&name);
    

void compute_bitfit_shadow(
        const core::PETLayerManager& pet_layer_manager,
        const int task_id,
        core::Tensor* task_input,
        core::Tensor* task_output, 
        core::Tensor* task_shadow_output,
        core::Tensor* task_hidden_states,
        core::Tensor* task_q_out,
        core::Tensor* task_k_out,
        core::Tensor* task_v_out,

        bool add_bias_act,  bool add_input_bias_layernorm, bool split_add_transpose, std::string&name);

void compute_adapter_shadow(const core::PETLayerManager& pet_layer_manager,
        const int task_id,
        core::Tensor* task_input,
        core::Tensor* task_output, 
        core::Tensor* task_shadow_output,
        core::Tensor* task_hidden_states,
        core::Tensor* task_q_out,
        core::Tensor* task_k_out,
        core::Tensor* task_v_out, 

        bool add_bias_act,  bool add_input_bias_layernorm, bool split_add_transpose, std::string&name);



void compute_nothing(const core::PETLayerManager& pet_layer_manager,
        const int task_id,
        core::Tensor* task_input,
        core::Tensor* task_output, 
        core::Tensor* task_shadow_output,
        core::Tensor* task_hidden_states,
        core::Tensor* task_q_out,
        core::Tensor* task_k_out,
        core::Tensor* task_v_out, 

        bool add_bias_act,  bool add_input_bias_layernorm, bool split_add_transpose, std::string&name);

shadow_op get_shadow_op(
  const core::PETLayerManager& pet_layer_manager,
  int task_id);

} // namespace layers
} // namespace turbo_transformers