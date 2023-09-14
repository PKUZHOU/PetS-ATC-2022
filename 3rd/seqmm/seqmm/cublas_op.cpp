#include "cublas_op.h"
#include <iostream>

namespace seqmm {

template<class T> void CuBlasMM<T>::Init(const float alpha, const float beta,
                                         const bool transA, const bool transB,
                                         const int m, const int n, const int k) {
  alpha_ = alpha;
  beta_ = beta;
  
  transA_ = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  transB_ = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  this->lda_ = transA ? m : k;
  this->ldb_ = transB ? k : n;
  this->ldc_ = n;

  if (sizeof(T) == sizeof(half)) {
    cuda_dtype_ = CUDA_R_16F;
  }
  
  CHECK_CUBLAS(cublasCreate(&handle_));
}

template<class T> void CuBlasMM<T>::Run(const T* d_mat_A, const T* d_mat_B,
                                        T* d_mat_C,
                                        const int m, const int n, const int k) {
  // Perform warmup operation with cublas.
  // Pay attention: as cublas store A, B and C in column-major order,
  // we compute (A*B)^T=(B^T)*(A^T)=C^T, instead of AB=C.
  CHECK_CUBLAS(cublasGemmEx(handle_, transB_, transA_,
                            n, m, k,
                            &alpha_,
                            d_mat_B, cuda_dtype_, this->ldb_,
                            d_mat_A, cuda_dtype_, this->lda_,
                            &beta_,
                            d_mat_C, cuda_dtype_, this->ldc_,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

template<class T> void CuBlasMM<T>::Clear() {
  CHECK_CUBLAS(cublasDestroy(handle_));
}

template void CuBlasMM<half>::Init(const float alpha, const float beta,
                                   const bool transA, const bool transB,
                                   const int m, const int n, const int k);
template void CuBlasMM<half>::Run(const half* d_mat_A, const half* d_mat_B,
                                   half* d_mat_C,
                                   const int m, const int n, const int k);
template void CuBlasMM<half>::Clear();

template void CuBlasMM<float>::Init(const float alpha, const float beta,
                                    const bool transA, const bool transB,
                                    const int m, const int n, const int k);
template void CuBlasMM<float>::Run(const float* d_mat_A, const float* d_mat_B,
                                   float* d_mat_C,
                                   const int m, const int n, const int k);
template void CuBlasMM<float>::Clear();

}  // namespace seqmm
