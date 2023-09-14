#include "sputnik_op.h"
#include "sputnik/cuda_spmm.h"
#include "utils.h"

#include <iostream>
#include <assert.h>

namespace seqmm {

void SputnikMM::Init() {
  CHECK_CUDA(cudaMalloc(&d_mat_A_trans_, A_size_ * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_mat_C_trans_, C_size_ * sizeof(float)));
}

void SputnikMM::Run(const float* d_mat_A,
                    const float* d_B_csr_values,
                    const int* d_B_csr_col_indices,
                    const int* d_B_csr_row_offsets,
                    const int* d_B_csr_row_indices,
                    const int nnz,
                    const int m, const int n, const int k,
                    float* d_mat_C,
                    bool trans,
                    cudaStream_t stream) {
  if (trans) {
    assert(A_size_ >= m * k);
    assert(C_size_ >= m * n);
        
    sputnik::CudaTranspose(d_mat_A, k, m, d_mat_A_trans_, stream);

    sputnik::CudaSpmm(n, k, m, nnz, d_B_csr_row_indices, d_B_csr_values,
                      d_B_csr_row_offsets, d_B_csr_col_indices,
                      d_mat_A_trans_, d_mat_C_trans_, stream);
  
    sputnik::CudaTranspose(d_mat_C_trans_, m, n, d_mat_C, stream);
  } else {
    sputnik::CudaSpmm(n, k, m, nnz, d_B_csr_row_indices, d_B_csr_values,
                      d_B_csr_row_offsets, d_B_csr_col_indices, d_mat_A, d_mat_C,
                      stream);
  }
}

void SputnikMM::Clear() {
  CHECK_CUDA(cudaFree(d_mat_A_trans_));
  CHECK_CUDA(cudaFree(d_mat_C_trans_));
}
  
}  // namespace seqmm
