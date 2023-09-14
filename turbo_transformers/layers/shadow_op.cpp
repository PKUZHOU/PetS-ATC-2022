#include "turbo_transformers/layers/shadow_op.h"
#include "turbo_transformers/layers/kernels/sparse_mat_mul.h"
#include "turbo_transformers/core/cuda_device_context.h"
#include "turbo_transformers/layers/kernels/utils.h"

namespace turbo_transformers {
namespace layers {

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

            bool add_bias_act,  bool add_input_bias_layernorm, bool split_add_transpose, std::string&name){
    // Compute the shadow output    
    core::Tensor * operand_A;

    if(task_hidden_states){
        operand_A = task_hidden_states;
    }
    else{
        operand_A = task_input;
    }

    auto cuda_ctx =
        turbo_transformers::core::CUDADeviceContext::GetInstance(task_id);

#ifdef SP_SHADOW
    std::shared_ptr<kernels::SparseMatMulCsr> sparse_matmul_ptr = pet_layer_manager.get_sparse_matmul(task_id);
    core::SparseTensor* sp_task_mask_ptr = pet_layer_manager.get_sp_mask_shadow(task_id);
    if (core::CUDADeviceContext::num_streams > 1) {
      sparse_matmul_ptr->Run(
          *operand_A, false,
          sp_task_mask_ptr, false,
          1.0, task_shadow_output, 0.0, cuda_ctx.get(), name + "SparseMatMul");
    } else {
      sparse_matmul_ptr->Run(
          *operand_A, false,
          sp_task_mask_ptr, false,
          1.0, task_shadow_output, 0.0, name + "SparseMatMul");
    }
#else
    // get the mask shadow
    const core::Tensor &task_mask_shadow = pet_layer_manager.get_maskbert_shadow(task_id); 
    //task shadow output
    layers::kernels::MatMul(*operand_A, false, task_mask_shadow, false, 1.0,
                            task_shadow_output, 0.0, name + "/MASK/MatMul");
#endif

    // Add to the dense output
    if (core::CUDADeviceContext::num_streams > 1) {
      layers::kernels::ElwsAdd(*task_shadow_output, *task_output, task_output,
                               cuda_ctx.get(),
                               name + "/MASK/ElwsAdd");
    } else {
      layers::kernels::ElwsAdd(*task_shadow_output, *task_output, task_output,
                               name + "/MASK/ElwsAdd");
    }

    const core::Tensor &task_bias = pet_layer_manager.get_bias(-1);
    if(add_input_bias_layernorm){
        const core::Tensor &task_layer_norm_bias = pet_layer_manager.get_layer_norm_bias(-1);
        const core::Tensor &task_layer_norm_weight = pet_layer_manager.get_layer_norm_weight(-1);
        if (core::CUDADeviceContext::num_streams > 1) {
          layers::kernels::AddBiasLayerNorm<float>(
              *task_input, task_bias, task_layer_norm_weight, task_layer_norm_bias,
              task_output, cuda_ctx.get(), 1e-12, name + "/MASK/AddBiasLayerNorm");
        } else {
          layers::kernels::AddBiasLayerNorm<float>(
              *task_input, task_bias, task_layer_norm_weight, task_layer_norm_bias,
              task_output, 1e-12, name + "/MASK/AddBiasLayerNorm");
        }
    }
    else if(add_bias_act){
      if (core::CUDADeviceContext::num_streams > 1) {
        layers::kernels::AddBiasAct<float, layers::kernels::ActivationType::Gelu>(
            task_bias, task_output, cuda_ctx.get(), name + "MASK/AddBiasAct");
      } else {
        layers::kernels::AddBiasAct<float, layers::kernels::ActivationType::Gelu>(
            task_bias, task_output, name + "MASK/AddBiasAct");
      }
    }
    else if(split_add_transpose){
      if (core::CUDADeviceContext::num_streams > 1) {
        layers::kernels::SplitAddBiasTransposeForScore(
            *task_output, task_bias, *task_q_out, *task_k_out, *task_v_out,
            cuda_ctx.get(),
            name + "/SplitAddBiasTransposeForScore");
      } else {
        layers::kernels::SplitAddBiasTransposeForScore(
            *task_output, task_bias, *task_q_out, *task_k_out, *task_v_out,
            name + "/SplitAddBiasTransposeForScore");
      }
    }
}

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

            bool add_bias_act,  bool add_input_bias_layernorm, bool split_add_transpose, std::string&name){
    // Compute the shadow output
  auto cuda_ctx =
      turbo_transformers::core::CUDADeviceContext::GetInstance(task_id);

    core::Tensor * operand_A;
    if(task_hidden_states){
        operand_A = task_hidden_states;
    }
    else{
        operand_A = task_input;
    }
#ifdef SP_SHADOW
    std::shared_ptr<kernels::SparseMatMul> sparse_matmul_ptr = pet_layer_manager.get_sparse_matmul(task_id);
    core::SparseTensor* sp_task_diff_ptr = pet_layer_manager.get_sp_diff_shadow(task_id);
    if (core::CUDADeviceContext::num_streams > 1) {
    sparse_matmul_ptr->Run(
        *operand_A, false,
        sp_task_diff_ptr, false,
        1.0, task_shadow_output, 0.0, cuda_ctx.get(), name + "SparseMatMul");
    } else {
      sparse_matmul_ptr->Run(
          *operand_A, false,
          sp_task_diff_ptr, false,
          1.0, task_shadow_output, 0.0, name + "SparseMatMul");
    }
#else 
    const core::Tensor &task_diff_shadow = pet_layer_manager.get_diff_shadow(task_id); // get the mask shadow  
    //task shadow output
    layers::kernels::MatMul(*operand_A, false, task_diff_shadow, false, 1.0,
            task_shadow_output, 0.0, name + "/DIFF/MatMul");
#endif
    // Add to the dense output
    const core::Tensor &task_bias = pet_layer_manager.get_bias(task_id);
    if (core::CUDADeviceContext::num_streams > 1) {
      layers::kernels::ElwsAdd(*task_shadow_output, *task_output, task_output,
                               cuda_ctx.get(),
                               name + "/DIFF/ElwsAdd");
    } else {
      layers::kernels::ElwsAdd(*task_shadow_output, *task_output, task_output,
                               name + "/DIFF/ElwsAdd");
    }
    if(add_input_bias_layernorm) {
        const core::Tensor &task_layer_norm_bias = pet_layer_manager.get_layer_norm_bias(task_id);
        const core::Tensor &task_layer_norm_weight = pet_layer_manager.get_layer_norm_weight(task_id);
        if (core::CUDADeviceContext::num_streams > 1) {
          layers::kernels::AddBiasLayerNorm<float>(
              *task_input, task_bias, task_layer_norm_weight, task_layer_norm_bias,
              task_output, cuda_ctx.get(), 1e-12, name + "/DIFF/AddBiasLayerNorm");
        } else {
          layers::kernels::AddBiasLayerNorm<float>(
              *task_input, task_bias, task_layer_norm_weight, task_layer_norm_bias,
              task_output, 1e-12, name + "/DIFF/AddBiasLayerNorm");
        }
    }
    else if(add_bias_act){
      if (core::CUDADeviceContext::num_streams > 1) {
        layers::kernels::AddBiasAct<float, layers::kernels::ActivationType::Gelu>(
            task_bias, task_output, cuda_ctx.get(), name + "DIFF/AddBiasAct");
      } else {
        layers::kernels::AddBiasAct<float, layers::kernels::ActivationType::Gelu>(
            task_bias, task_output, name + "DIFF/AddBiasAct");
      }
    }
    else if(split_add_transpose){
      if (core::CUDADeviceContext::num_streams > 1) {
       layers::kernels::SplitAddBiasTransposeForScore(*task_output, task_bias, *task_q_out,
                                                     *task_k_out, *task_v_out,
                                                     cuda_ctx.get(),
                                                     name + "/SplitAddBiasTransposeForScore");
      } else {
      layers::kernels::SplitAddBiasTransposeForScore(*task_output, task_bias, *task_q_out,
                                                     *task_k_out, *task_v_out,
                                                     name + "/SplitAddBiasTransposeForScore");
      }
    }
}

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
        bool add_bias_act, bool add_input_bias_layernorm,
        bool split_add_transpose, std::string&name){
  auto cuda_ctx =
      turbo_transformers::core::CUDADeviceContext::GetInstance(task_id);

  const core::Tensor &task_bias = pet_layer_manager.get_bias(task_id);
  if(add_input_bias_layernorm){
    const core::Tensor &task_layer_norm_bias = pet_layer_manager.get_layer_norm_bias(task_id);
    const core::Tensor &task_layer_norm_weight = pet_layer_manager.get_layer_norm_weight(task_id);
    if (core::CUDADeviceContext::num_streams > 1) {
      layers::kernels::AddBiasLayerNorm<float>(
          *task_input, task_bias, task_layer_norm_weight, task_layer_norm_bias,
          task_output, cuda_ctx.get(), 1e-12, name + "/BITFIT/AddBiasLayerNorm");
    } else {
      layers::kernels::AddBiasLayerNorm<float>(
          *task_input, task_bias, task_layer_norm_weight, task_layer_norm_bias,
          task_output, 1e-12, name + "/BITFIT/AddBiasLayerNorm");
    }
  }
  else if(add_bias_act) {
    if (core::CUDADeviceContext::num_streams > 1) {
      layers::kernels::AddBiasAct<float, layers::kernels::ActivationType::Gelu>(
          task_bias, task_output, cuda_ctx.get(), name + "BITFIT/AddBiasAct");
    } else {
      layers::kernels::AddBiasAct<float, layers::kernels::ActivationType::Gelu>(
          task_bias, task_output, name + "BITFIT/AddBiasAct");
    }
  }
  else if(split_add_transpose){
    if (core::CUDADeviceContext::num_streams > 1) {
      layers::kernels::SplitAddBiasTransposeForScore(*task_output, task_bias, *task_q_out,
                                                     *task_k_out, *task_v_out,
                                                     cuda_ctx.get(),
                                                     name + "/SplitAddBiasTransposeForScore");
    } else {
      layers::kernels::SplitAddBiasTransposeForScore(*task_output, task_bias, *task_q_out,
                                                     *task_k_out, *task_v_out,
                                                     name + "/SplitAddBiasTransposeForScore");
    }
  }
}

void compute_adapter_shadow(
        const core::PETLayerManager& pet_layer_manager,
        const int task_id,
        core::Tensor* task_input,
        core::Tensor* task_output, 
        core::Tensor* task_shadow_output,
        core::Tensor* task_hidden_states,
        core::Tensor* task_q_out,
        core::Tensor* task_k_out,
        core::Tensor* task_v_out, 
        bool add_bias_act, bool add_input_bias_layernorm,
        bool split_add_transpose, std::string&name){
  auto cuda_ctx =
      turbo_transformers::core::CUDADeviceContext::GetInstance(task_id);

  const core::Tensor &down_scale_w = pet_layer_manager.get_adapter_params(task_id, true, true);
  const core::Tensor &down_scale_b = pet_layer_manager.get_adapter_params(task_id, true, false);
  const core::Tensor &up_scale_w = pet_layer_manager.get_adapter_params(task_id, false, true);
  const core::Tensor &up_scale_b = pet_layer_manager.get_adapter_params(task_id, false, false);
    
  core::Tensor intermeidate_buffer(nullptr); // declare the shadow output for current task
  if (core::CUDADeviceContext::num_streams > 1) {
    intermeidate_buffer.Reshape<float>({task_input->shape(0), task_input->shape(1), down_scale_b.shape(0)},
                                       task_input->device_type(), task_input->device_id(),
                                       cuda_ctx.get(),
                                       "Adapter/Reshape");
  } else {
    intermeidate_buffer.Reshape<float>({task_input->shape(0), task_input->shape(1), down_scale_b.shape(0)},
                                       task_input->device_type(), task_input->device_id(),
                                       "Adapter/Reshape");
  }
  
  //input + shared bias  # assume this bias has been merged into the next linear layer
  // const core::Tensor &task_bias = pet_layer_manager.get_bias(-1);
  // kernels::AddBias(
  //         task_bias, task_output, name + "Adapter/AddBias");

  //Different from the DIFF and MASK. Adapter computes based on the dense outputs. 
  //Downscale
  if (core::CUDADeviceContext::num_streams > 1) {
    layers::kernels::MatMul(*task_output, false, down_scale_w, false, 1.0,
                            &intermeidate_buffer, 0.0,
                            cuda_ctx.get(),
                            name + "/Adapter_downscale/MatMul");
  } else {
    layers::kernels::MatMul(*task_output, false, down_scale_w, false, 1.0,
                            &intermeidate_buffer, 0.0,
                            name + "/Adapter_downscale/MatMul");
  }
  
  //Activation and bias
  if (core::CUDADeviceContext::num_streams > 1) {
    layers::kernels::AddBiasAct<float, layers::kernels::ActivationType::Gelu>(
        down_scale_b, &intermeidate_buffer, cuda_ctx.get(),
        name + "Adapter_downscale/AddBiasAct");
  } else {
    layers::kernels::AddBiasAct<float, layers::kernels::ActivationType::Gelu>(
        down_scale_b, &intermeidate_buffer,
        name + "Adapter_downscale/AddBiasAct");
  }
  //upscale
  if (core::CUDADeviceContext::num_streams > 1) {
    layers::kernels::MatMul(intermeidate_buffer, false, up_scale_w, false, 1.0,
                            task_shadow_output, 0.0, cuda_ctx.get(),
                            name + "/Adapter_upscale/MatMul");
  } else {
    layers::kernels::MatMul(intermeidate_buffer, false, up_scale_w, false, 1.0,
                            task_shadow_output, 0.0,
                            name + "/Adapter_upscale/MatMul");
  }
  //residual connection
  if (core::CUDADeviceContext::num_streams > 1) {
    layers::kernels::ElwsAdd(*task_shadow_output, *task_output, task_output,
                             cuda_ctx.get(),
                             name + "/Adapter_upscale/ElwsAdd");
  } else {
    layers::kernels::ElwsAdd(*task_shadow_output, *task_output, task_output,
                             name + "/Adapter_upscale/ElwsAdd");
  }
  
  if(add_input_bias_layernorm){
    const core::Tensor &task_layer_norm_bias = pet_layer_manager.get_layer_norm_bias(task_id);
    const core::Tensor &task_layer_norm_weight = pet_layer_manager.get_layer_norm_weight(task_id);
    if (core::CUDADeviceContext::num_streams > 1) {
      layers::kernels::AddBiasLayerNorm<float>(
          *task_input, up_scale_b, task_layer_norm_weight, task_layer_norm_bias,
          task_output, cuda_ctx.get(), 1e-12, name + "/Adapter/AddBiasLayerNorm");
    } else {
      layers::kernels::AddBiasLayerNorm<float>(
          *task_input, up_scale_b, task_layer_norm_weight, task_layer_norm_bias,
          task_output, 1e-12, name + "/Adapter/AddBiasLayerNorm");
    }
  }
  else if(add_bias_act){
    std::cerr << "Invalid operation" << std::endl;

  }
  else if(split_add_transpose){
    std::cerr << "Invalid operation" << std::endl;
  }
}

void compute_nothing(const core::PETLayerManager& pet_layer_manager,
                     const int task_id,
                     core::Tensor* task_input,
                     core::Tensor* task_output, 
                     core::Tensor* task_shadow_output,
                     core::Tensor* task_hidden_states,
                     core::Tensor* task_q_out,
                     core::Tensor* task_k_out,
                     core::Tensor* task_v_out, 
                     bool add_bias_act,  bool add_input_bias_layernorm,
                     bool split_add_transpose, std::string&name) {
  auto cuda_ctx =
        turbo_transformers::core::CUDADeviceContext::GetInstance(task_id);

  const core::Tensor &task_bias = pet_layer_manager.get_bias(-1);
  if(add_input_bias_layernorm){
      const core::Tensor &task_layer_norm_bias = pet_layer_manager.get_layer_norm_bias(-1);
      const core::Tensor &task_layer_norm_weight = pet_layer_manager.get_layer_norm_weight(-1);
      if (core::CUDADeviceContext::num_streams > 1) {
        layers::kernels::AddBiasLayerNorm<float>(
            *task_input, task_bias, task_layer_norm_weight, task_layer_norm_bias,
            task_output, cuda_ctx.get(), 1e-12, name + "/Nothing/AddBiasLayerNorm");
      } else {
        layers::kernels::AddBiasLayerNorm<float>(
            *task_input, task_bias, task_layer_norm_weight, task_layer_norm_bias,
            task_output, 1e-12, name + "/Nothing/AddBiasLayerNorm");
      }
  }
  else if(add_bias_act){
    if (core::CUDADeviceContext::num_streams > 1) {
      layers::kernels::AddBiasAct<float, layers::kernels::ActivationType::Gelu>(
          task_bias, task_output, cuda_ctx.get(), name + "Nothing/AddBiasAct");
    } else {
      layers::kernels::AddBiasAct<float, layers::kernels::ActivationType::Gelu>(
          task_bias, task_output, name + "Nothing/AddBiasAct");
    }
  }
  else if(split_add_transpose){
    if (core::CUDADeviceContext::num_streams > 1) {
      layers::kernels::SplitAddBiasTransposeForScore(
          *task_output, task_bias, *task_q_out, *task_k_out, *task_v_out,
          cuda_ctx.get(),
          name + "/SplitAddBiasTransposeForScore");
    } else {
      layers::kernels::SplitAddBiasTransposeForScore(
          *task_output, task_bias, *task_q_out, *task_k_out, *task_v_out,
          name + "/SplitAddBiasTransposeForScore");
    }
  }

};

shadow_op get_shadow_op(
  const core::PETLayerManager& pet_layer_manager,
  int task_id){
    int pet_type = pet_layer_manager.get_pet_type(task_id);

    switch (pet_type)
    {
    case MASK_BERT:
        return compute_mask_shadow;
    case DIFF_PRUNING:
        return compute_diff_shadow;
    case BITFIT:
        return compute_bitfit_shadow;
    case ADAPTERS:
        return compute_adapter_shadow;
    case STANDARD:
        return compute_nothing;
    default:
        std::cerr<<"Unsupported Shadow Operation!"<<std::endl;
        break;
    }
}

} // namespace layers
} // namespace turbo_transformers
