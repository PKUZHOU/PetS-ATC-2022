
#include "paisparse_op.h"
#include "utils.h"

#include "sparse_kernels.hxx"

namespace seqmm {

template<class T> void PaiSparseMM<T>::Run(const T* d_mat_A,
                                           const T* d_B_csr_values,
                                           const int* d_B_csr_col_indices,
                                           const int* d_B_csr_row_offsets,
                                           const int nnz,
                                           const int m, const int n, const int k,
                                           const bool transA, const bool transB,
                                           const T alpha, const T beta,
                                           T* d_mat_C,
                                           cudaStream_t stream) {
  
  cusparseOperation_t co_transA = transA ? CUSPARSE_OPERATION_TRANSPOSE :
      CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t co_transB = transB ? CUSPARSE_OPERATION_TRANSPOSE :
      CUSPARSE_OPERATION_NON_TRANSPOSE;

  int lda = transA ? m : k;
  int ldb = transB ? n : k;
  int ldc = n;

  if (1 == m) {
    CHECK_CUSPARSE(SparseMV(co_transA,
                            n, m,
                            nnz, &alpha,
                            d_B_csr_values,
                            d_B_csr_row_offsets,
                            d_B_csr_col_indices,
                            d_mat_A, &beta,
                            d_mat_C,
                            (cudaStream_t* )&(stream)));
  } else {
    CHECK_CUSPARSE(SparseMM(co_transA, co_transB,
                            n, m, k,
                            nnz, &alpha,
                            d_B_csr_values,
                            d_B_csr_row_offsets,
                            d_B_csr_col_indices,
                            d_mat_A, lda, &beta,
                            d_mat_C, ldc,
                            (cudaStream_t* )&(stream)));
  }
}

template void PaiSparseMM<float>::Run(const float* d_mat_A,
                                      const float* d_B_csr_values,
                                      const int* d_B_csr_col_indices,
                                      const int* d_B_csr_row_offsets,
                                      const int nnz,
                                      const int m, const int n, const int k,
                                      const bool transA, const bool transB,
                                      const float alpha, const float beta,
                                      float* d_mat_C,
                                      cudaStream_t stream);

template void PaiSparseMM<half>::Run(const half* d_mat_A,
                                     const half* d_B_csr_values,
                                     const int* d_B_csr_col_indices,
                                     const int* d_B_csr_row_offsets,
                                     const int nnz,
                                     const int m, const int n, const int k,
                                     const bool transA, const bool transB,
                                     const half alpha, const half beta,
                                     half* d_mat_C,
                                     cudaStream_t stream);

}  // namespace seqmm
