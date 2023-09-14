#ifndef SEQMM_CUBLAS_OP_H_
#define SEQMM_CUBLAS_OP_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "dense_op_base.h"
#include "utils.h"

namespace seqmm {

template<class T>
class CuBlasMM : public MatMul<T> {
 public:
  void Init(const float alpha, const float beta,
            const bool transA, const bool transB,
            const int m, const int n, const int k);
  
  void Run(const T* d_mat_A, const T* d_mat_B, T* d_mat_C,
           const int m, const int n, const int k);
  
  void Clear();

private:
  cublasOperation_t transA_;
  cublasOperation_t transB_;

  float alpha_;
  float beta_;

  cudaDataType cuda_dtype_ = CUDA_R_32F;

  cublasHandle_t handle_;
};  // class MatMul

}  // namespace seqmm

#endif  // SEQMM_DENSE_OP_H_
