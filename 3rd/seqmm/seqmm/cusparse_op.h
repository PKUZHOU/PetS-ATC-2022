#ifndef SEQMM_CUSPARSE_H_
#define SEQMM_CUSPARSE_H_

#include <stdio.h>
#include <cusparse_v2.h>

namespace seqmm {

template<class T>
class CuSparseMM {
 public:
  void Init();

  void Run(const T* d_mat_A,
           const T* d_B_csr_values,
           const int* d_B_csr_col_indices,
           const int* d_B_csr_row_offsets,
           const int nnz,
           const int m, const int n, const int k,
           const bool transA, const bool transB,
           const float alpha, const float beta,
           T* d_mat_C,
           cusparseHandle_t cusparse_handle);

  void Clear();
  
 private:
  cusparseDnMatDescr_t mat_B_dense_descr_;
  cusparseDnMatDescr_t mat_A_descr_, mat_C_descr_;  // for SpMM
  cusparseDnVecDescr_t vec_A_descr_, vec_C_descr_;  // for SpMV
  cusparseSpMatDescr_t mat_B_sparse_descr_;

  void* cusparse_buffer_ = nullptr;
  size_t cusparse_buffer_size_ = 0;

  cusparseSpMMAlg_t spmm_alg_ = CUSPARSE_SPMM_ALG_DEFAULT;
  cusparseSpMVAlg_t spmv_alg_ = CUSPARSE_SPMV_ALG_DEFAULT;
};  // class CuSparseMM

}  // namespace seqmm

#endif  // SEQMM_CUSPARSE_H_
