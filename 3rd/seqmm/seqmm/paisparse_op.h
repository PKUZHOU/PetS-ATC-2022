#ifndef SEQMM_PAISPARSE_H_
#define SEQMM_PAISPARSE_H_

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <iostream>

namespace seqmm {

template<class T>
class PaiSparseMM {
 public:
  void Run(const T* d_mat_A,
           const T* d_B_csr_values,
           const int* d_B_csr_col_indices,
           const int* d_B_csr_row_offsets,
           const int nnz,
           const int m, const int n, const int k,
           const bool transA, const bool transB,
           const T alpha, const T beta,
           T* d_mat_C,
           cudaStream_t stream);
  
};  // class PaiSparseMM

}  // namespace seqmm

#endif  // SEQMM_PAISPARSE_H_
