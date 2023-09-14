#ifndef SEQMM_SPUTNIK_H_
#define SEQMM_SPUTNIK_H_

#include <cuda_runtime.h>

namespace seqmm {

class SputnikMM {
 public:
  void Init();
  
  void Run(const float* d_mat_A,
           const float* d_B_csr_values,
           const int* d_B_csr_col_indices,
           const int* d_B_csr_row_offsets,
           const int* d_B_csr_row_indices,
           const int nnz,
           const int m, const int n, const int k,
           float* d_mat_C,
           bool trans,
           cudaStream_t stream);

  void Clear();

 private:
  int A_size_ = 4 * 1024 * 1024;  // m*k, 4MB
  int C_size_ = 4 * 1024 * 1024;  // m*n, 4MB
  float* d_mat_A_trans_ = nullptr;
  float* d_mat_C_trans_ = nullptr;
  
};  // class SputnikMM

}  // namespace seqmm

#endif  // SEQMM_SPUTNIK_H_
