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

#ifdef TT_WITH_CUDA
#include "turbo_transformers/core/cuda_device_context.h"
#endif
#include "catch2/catch.hpp"

#include <iostream>

namespace turbo_transformers {
namespace core {

#ifdef TT_WITH_CUDA
TEST_CASE("CUDADeviceContext", "init") {
  auto& cuda_ctx0 = CUDADeviceContext::GetInstance();
  std::cout << "Stream of CUDA context: " << cuda_ctx0.stream() << std::endl;
  cuda_ctx0.Wait();

  auto& cuda_ctx1 = CUDADeviceContext::GetInstance();
  std::cout << "Stream of CUDA context: " << cuda_ctx1.stream() << std::endl;

  auto cuda_ctx_ptr0 = CUDADeviceContext::GetInstance(0);
  std::cout << "Stream of CUDA context#0: " << cuda_ctx_ptr0->stream() << std::endl;

  auto cuda_ctx_ptr1 = CUDADeviceContext::GetInstance(1);
  std::cout << "Stream of CUDA context#1: " << cuda_ctx_ptr1->stream() << std::endl;

  auto cuda_ctx_ptr00 = CUDADeviceContext::GetInstance(0);
  std::cout << "Stream of CUDA context#0: " << cuda_ctx_ptr00->stream() << std::endl;

  CUDADeviceContext::num_streams = 5;

  cuda_ctx1.Destroy();
  cuda_ctx_ptr0->Destroy();
  cuda_ctx_ptr1->Destroy();
  cuda_ctx_ptr00->Destroy();
}

#endif

}  // namespace core
}  // namespace turbo_transformers
