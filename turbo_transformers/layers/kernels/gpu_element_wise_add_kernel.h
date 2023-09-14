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

namespace turbo_transformers {
namespace layers {
namespace kernels {

void GPUElwsAdd(const float* A, const float * B, float* output, const  int64_t batch_size,  const int64_t feature_dim, cudaStream_t stream);
    
}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
