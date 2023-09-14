#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <iostream>

#include "cusparse_op.h"
#include "utils.h"

namespace seqmm {

template<class T> void CuSparseMM<T>::Init() {

}

template<class T> void CuSparseMM<T>::Run(const T* d_mat_A,
                                          const T* d_B_csr_values,
                                          const int* d_B_csr_col_indices,
                                          const int* d_B_csr_row_offsets,
                                          const int nnz,
                                          const int m, const int n, const int k,
                                          const bool transA, const bool transB,
                                          const float alpha, const float beta,
                                          T* d_mat_C,
                                          cusparseHandle_t cusparse_handle) {
  int row_B = transB ? k : n;
  int col_B = transB ? n : k;

  cusparseOperation_t co_transA = transA ? CUSPARSE_OPERATION_TRANSPOSE :
      CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t co_transB = transB ? CUSPARSE_OPERATION_TRANSPOSE :
      CUSPARSE_OPERATION_NON_TRANSPOSE;
  
  int lda = transA ? m : k;
  int ldb = transB ? n : k;
  int ldc = n;

  // TODO: support more types and error out with unsupported types.
  cudaDataType cuda_dtype = CUDA_R_32F;
  if (sizeof(T) == sizeof(half)) {
    cuda_dtype = CUDA_R_16F;
  }

  CHECK_CUSPARSE(cusparseCreateCsr(&mat_B_sparse_descr_, row_B, col_B,
                                    nnz,
                                    (void*)d_B_csr_row_offsets,
                                    (void*)d_B_csr_col_indices,
                                    (void*)d_B_csr_values,
                                    CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_32I, 
                                    CUSPARSE_INDEX_BASE_ZERO,
                                    cuda_dtype));

  if (1 == m) {
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_A_descr_, k, (void*)d_mat_A,
                                       cuda_dtype));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_C_descr_, n, d_mat_C,
                                       cuda_dtype));
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparse_handle,
                                           co_transB,
                                           &alpha, mat_B_sparse_descr_,
                                           vec_A_descr_,
                                           &beta, vec_C_descr_, CUDA_R_32F,
                                           spmv_alg_,
                                           &cusparse_buffer_size_));
  } else {
    CHECK_CUSPARSE( cusparseCreateDnMat(&mat_A_descr_, k, m, lda,
                                        (void*)d_mat_A,
                                        cuda_dtype, CUSPARSE_ORDER_COL) );
    CHECK_CUSPARSE( cusparseCreateDnMat(&mat_C_descr_, n, m, ldc,
                                        d_mat_C,
                                        cuda_dtype, CUSPARSE_ORDER_COL) );
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
        cusparse_handle,
        co_transB, co_transA,
        &alpha, mat_B_sparse_descr_, mat_A_descr_,
        &beta, mat_C_descr_,
        CUDA_R_32F, spmm_alg_, &cusparse_buffer_size_) );
  }

  CHECK_CUDA(cudaMalloc(&cusparse_buffer_, cusparse_buffer_size_));

  if (1 == m) {
    CHECK_CUSPARSE(cusparseSpMV(cusparse_handle,
                                co_transB, &alpha,
                                mat_B_sparse_descr_, vec_A_descr_,
                                &beta, vec_C_descr_, CUDA_R_32F, spmv_alg_,
                                cusparse_buffer_));
  } else {
    CHECK_CUSPARSE(cusparseSpMM(cusparse_handle,
                                co_transB,
                                co_transA,
                                &alpha, mat_B_sparse_descr_, mat_A_descr_,
                                &beta,
                                mat_C_descr_, CUDA_R_32F,
                                spmm_alg_, cusparse_buffer_));
  }

  if (1 == m) {
    CHECK_CUSPARSE(cusparseDestroyDnVec(vec_A_descr_));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vec_C_descr_));
  } else {
    CHECK_CUSPARSE(cusparseDestroyDnMat(mat_A_descr_));
    CHECK_CUSPARSE(cusparseDestroyDnMat(mat_C_descr_));
  }
  CHECK_CUSPARSE(cusparseDestroyDnMat(mat_B_dense_descr_));
  CHECK_CUSPARSE(cusparseDestroySpMat(mat_B_sparse_descr_));
  
  CHECK_CUDA(cudaFree(cusparse_buffer_));
}

template<class T> void CuSparseMM<T>::Clear() {
  //  std::cout << "I am CusparseMM::Clear()" << std::endl;
}

template void CuSparseMM<half>::Init();
template void CuSparseMM<float>::Init();

template void CuSparseMM<half>::Run(const half* d_mat_A,
                                    const half* d_B_csr_values,
                                    const int* d_B_csr_col_indices,
                                    const int* d_B_csr_row_offsets,
                                    const int nnz,
                                    const int m, const int n, const int k,
                                    const bool transA, const bool transB,
                                    const float alpha, const float beta,
                                    half* d_mat_C,
                                    cusparseHandle_t cusparse_handle);
template void CuSparseMM<float>::Run(const float* d_mat_A,
                                     const float* d_B_csr_values,
                                     const int* d_B_csr_col_indices,
                                     const int* d_B_csr_row_offsets,
                                     const int nnz,
                                     const int m, const int n, const int k,
                                     const bool transA, const bool transB,
                                     const float alpha, const float beta,
                                     float* d_mat_C,
                                     cusparseHandle_t cusparse_handle);

template void CuSparseMM<half>::Clear();
template void CuSparseMM<float>::Clear();

}  // namespace seqmm
