
#include "catch2/catch.hpp"
#include "turbo_transformers/layers/kernels/sparse_mat_mul.h"
#include "turbo_transformers/core/tensor_copy.h"
#include "turbo_transformers/layers/kernels/common.h"

#include <iostream>

namespace turbo_transformers {
namespace layers {
namespace kernels {

TEST_CASE("SparseMatMulTest", "[CSR]") {
  turbo_transformers::core::Tensor cpu_A(
      turbo_transformers::core::NewDLPackTensorT<float>({2, 2}));
  float *A_data = cpu_A.mutableData<float>();
  for (int i = 0; i < 4; ++i) A_data[i] = i * 1.0;
  
  turbo_transformers::core::Tensor gpu_A =
      common::CreateTensor<float>({2, 2}, kDLGPU, 0);

  core::Copy<float>(cpu_A, gpu_A);

  turbo_transformers::core::Tensor gpu_C =
      common::CreateTensor<float>({2, 2}, kDLGPU, 0);
  turbo_transformers::core::Tensor cpu_C =
      common::CreateTensor<float>({2, 2}, kDLCPU, 0);

  
  turbo_transformers::core::Tensor cpu_B(
      turbo_transformers::core::NewDLPackTensorT<float>({2, 2}));
  float *B_data = cpu_B.mutableData<float>();

  /*
   * B_data (50% sparsity):
   * 1.0 0.0
   * 0.0 1.0
   */
  B_data[0] = 1.0;
  B_data[1] = 0.0;
  B_data[2] = 0.0;
  B_data[3] = 1.0;
  
  turbo_transformers::core::SparseTensorCsr<float> sparse_B(cpu_B);

  sparse_B.Dense2Sparse();

  // Multiplication.
  std::shared_ptr<SparseMatMul> sparse_matmul_ptr =
      SparseMatMul::SparseMatMulFactory(seqmm::SparseFormat::kFmtCSR);

  sparse_matmul_ptr->Init();
  sparse_matmul_ptr->Run(
      gpu_A, false,
      &sparse_B, false,
      1.0, &gpu_C, 0.0, 0, "SparseMatMulCsrTest");
  sparse_matmul_ptr->Clear();

  core::Copy<float>(gpu_C, cpu_C);

  for (int i = 0; i < 4; ++i) {
    std::cout << cpu_C.data<float>()[i] << std::endl;
    REQUIRE(fabs(cpu_C.data<float>()[i] - i * 1.0) < 1e-6);
  }

  // Clear.
  sparse_B.Destroy();
}

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
