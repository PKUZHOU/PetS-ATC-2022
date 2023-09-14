#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <numeric>

#include "turbo_transformers/layers/kernels/gpu_element_wise_add_kernel.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {

static __global__ void element_add(const float* A, const float* B,
                                    float* out, const int batch_size, const int feature_dim) {
  float val;

  int row_id;
  int elem_per_thread = (feature_dim + blockDim.x - 1) / blockDim.x;
  int tid = threadIdx.x;

  for (int i = 0; i < elem_per_thread; ++i) {
    int offset = i * blockDim.x + tid;
    if (offset < feature_dim) {
      row_id = blockIdx.x;
      val = A[offset + row_id * feature_dim] + B[offset + row_id * feature_dim];
      out[offset + row_id * feature_dim] = val;
    }
  }
}
void GPUElwsAdd(const float* A,  const float* B, float* output,  int64_t batch_size, int64_t feature_dim, cudaStream_t stream){
    dim3 grid(batch_size); // minibatch
    int block_size = min(1024, (int)(feature_dim / 4));
    dim3 block(block_size);
    element_add<<<grid, block, 0, stream>>>(A, B, output, batch_size,
                                                       feature_dim);
}

}
}
}


