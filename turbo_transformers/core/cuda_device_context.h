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

#pragma once
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cuda_runtime.h>
#include <memory.h>

#include <vector>
#include <memory>

#include <map>

#include "macros.h"

namespace turbo_transformers {
namespace core {

void SHADOW_SYNC_PROLOG();
void SHADOW_SYNC_EPILOG();

class CUDADeviceContext {
 public:
  CUDADeviceContext();

  ~CUDADeviceContext();

  static CUDADeviceContext& GetInstance() {
    static CUDADeviceContext instance;
    return instance;
  }

  static std::shared_ptr<CUDADeviceContext> GetInstance(int n) {
    // FIXME: GetInstance for the nth task only works when tasks call
    // me in sequential order.
    int stream_id = (n % num_streams);


    while (instances.size() <= stream_id)
    {
      std::shared_ptr<CUDADeviceContext> instance =
        std::make_shared<CUDADeviceContext>();
      instances.push_back(instance);
    }
    
      return instances.at(stream_id);
  }  


  static void SyncAllInstances() {
    for (auto& instance : instances) {
      instance->Wait();
    }
  }

  void Wait() const;

  void Destroy();

  cudaStream_t stream() const;

  cublasHandle_t cublas_handle() const;

  cusparseHandle_t cusparse_handle() const;

  int compute_major() const;

 public:
  static int num_streams;


 private:
  static std::vector<std::shared_ptr<CUDADeviceContext>> instances;

  cudaStream_t stream_;
  cublasHandle_t handle_;
  cusparseHandle_t sp_handle_;
  cudaDeviceProp device_prop_;
  DISABLE_COPY_AND_ASSIGN(CUDADeviceContext);
};

}  // namespace core
}  // namespace turbo_transformers
