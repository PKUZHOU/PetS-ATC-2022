#include "turbo_transformers/layers/kernels/elementwise_add.h"
// #define TT_WITH_CUDA
#include "common.h"
#include "turbo_transformers/layers/kernels/common.h"
#ifdef TT_WITH_CUDA
#include "turbo_transformers/core/cuda_device_context.h"
#include "turbo_transformers/layers/kernels/common.h"
#include "turbo_transformers/layers/kernels/gpu_element_wise_add_kernel.h"
#endif
#ifdef WITH_PERFTOOLS
#include "turbo_transformers/core/profiler.h"
#endif

namespace turbo_transformers {
namespace layers {
namespace kernels {

void ElwsAdd(const core::Tensor& A, const core::Tensor& B, core::Tensor* out,
             core::CUDADeviceContext* cuda_ctx_ptr,
             const std::string name){

  int64_t feature_dim = A.shape(-1);
  int64_t batch_size = A.numel()/feature_dim; // minibatch * L

  if (A.device_type() == kDLCPU && B.device_type() == kDLCPU &&
      out->device_type() == kDLCPU) {
    TT_THROW("Only support GPU.");
  } 
  else if (A.device_type() == kDLGPU && B.device_type() == kDLGPU &&
           out->device_type() == kDLGPU) {
    const float* A_tensor = A.data<float>();
    const float* B_tensor = B.data<float>();
    float* out_tensor = out->mutableData<float>();
#ifdef TT_WITH_CUDA
    auto& cuda_ctx = (nullptr == cuda_ctx_ptr) ?
        core::CUDADeviceContext::GetInstance() : *cuda_ctx_ptr;
    
    GPUElwsAdd(A_tensor, B_tensor, out_tensor, batch_size, feature_dim, cuda_ctx.stream());
#else
    TT_THROW("The current code is not compiled with CUDA.");
#endif
  }
  else {
    TT_THROW("device_type %d is not supported for Elementwise add", A.device_type());
  }
}

void ElwsAdd(const core::Tensor& A, const core::Tensor& B, core::Tensor* out,
             const std::string name) {
  ElwsAdd(A, B, out, nullptr, name);
}

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
