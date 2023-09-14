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

#include "turbo_transformers/core/cuda_device_context.h"

#include "turbo_transformers/core/cuda_enforce.cuh"
#include "turbo_transformers/core/enforce.h"
#include "turbo_transformers/core/memory.h"
#include "turbo_transformers/core/cuda_utils.h"
namespace turbo_transformers {
namespace core {

void SHADOW_SYNC_PROLOG() {
#ifdef USE_NVTX
  auto& single_cuda_ctx =
      CUDADeviceContext::GetInstance();
  single_cuda_ctx.Wait();
#else
  if (CUDADeviceContext::num_streams != 1) {
    auto& single_cuda_ctx =
        CUDADeviceContext::GetInstance();
    single_cuda_ctx.Wait();
  }
#endif
}

void SHADOW_SYNC_EPILOG() {
  if (CUDADeviceContext::num_streams > 1) {
    CUDADeviceContext::SyncAllInstances();
    // cudaDeviceSynchronize();
  }
#ifdef USE_NVTX
  if (1 == CUDADeviceContext::num_streams) {
    auto& single_cuda_ctx =
        CUDADeviceContext::GetInstance();
    single_cuda_ctx.Wait();
  }
#endif
}


CUDADeviceContext::CUDADeviceContext() {
  TT_ENFORCE_CUDA_SUCCESS(cudaStreamCreate(&stream_));
  TT_ENFORCE_CUDA_SUCCESS(cublasCreate(&handle_));
  cusparseCreate(&sp_handle_);
  TT_ENFORCE_CUDA_SUCCESS(cublasSetStream(handle_, stream_));
  cusparseSetStream(sp_handle_, stream_);
  TT_ENFORCE_CUDA_SUCCESS(cudaGetDeviceProperties(&device_prop_, 0));
}

void CUDADeviceContext::Wait() const {
  cudaError_t e_sync = cudaSuccess;
  e_sync = cudaStreamSynchronize(stream_);
  TT_ENFORCE_CUDA_SUCCESS(e_sync);
}

cudaStream_t CUDADeviceContext::stream() const { return stream_; }

cublasHandle_t CUDADeviceContext::cublas_handle() const { return handle_; }

cusparseHandle_t CUDADeviceContext::cusparse_handle() const { return sp_handle_; }

int CUDADeviceContext::compute_major() const { return device_prop_.major; }

void CUDADeviceContext::Destroy() {
  Wait();
  if (nullptr != handle_) {
    TT_ENFORCE_CUDA_SUCCESS(cublasDestroy(handle_));
    handle_ = nullptr;
  }

  if (nullptr != sp_handle_) {
    cusparseDestroy(sp_handle_);
    sp_handle_ = nullptr;
  }
  
  if (nullptr != stream_) {
    TT_ENFORCE_CUDA_SUCCESS(cudaStreamDestroy(stream_));
    stream_ = nullptr;
  }
}

CUDADeviceContext::~CUDADeviceContext() {
 Wait();
 TT_ENFORCE_CUDA_SUCCESS(cublasDestroy(handle_));
 cusparseDestroy(sp_handle_);
 TT_ENFORCE_CUDA_SUCCESS(cudaStreamDestroy(stream_));

  // Do not do resource deallocation in destroctor of singleton managing
  // GPU resources. Becauase the singleton destructor is called after the
  // program exits, when CUDA has already been shutdown.
  // Details can be referred to:
  // https://stackoverflow.com/questions/35815597/cuda-call-fails-in-destructor.
}

std::vector<std::shared_ptr<CUDADeviceContext>> CUDADeviceContext::instances;
int CUDADeviceContext::num_streams = 1;

}  // namespace core
}  // namespace turbo_transformers
