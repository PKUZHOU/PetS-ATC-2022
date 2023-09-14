#include "sparse_tensor.h"
#include "cuda_enforce.cuh"
#include "tensor_copy.h"

#include "seqmm/utils.h"
// #include "seqmm/cusparse_op.h"

#include <iostream>

namespace turbo_transformers {

namespace core {

template<class T> void SparseTensorCsr<T>::Dense2Sparse() {
  assert(2 == dense_.n_dim());

  T* matrix;
#if IF_TRANS == 1
  T* matrix_trans;
#endif

  int64_t matrix_size = dense_.shape(0) * dense_.shape(1);

 #if IF_TRANS == 1
  if (kDLGPU == dense_.device_type()) {
    matrix_trans = new T[matrix_size];
    Copy<T>(dense_.data<T>(), matrix_size, kDLGPU, kDLCPU, matrix_trans);
  } else {
    matrix_trans = (T* )dense_.data<T>();
  }
  matrix = new T[matrix_size];
  seqmm::Transpose(matrix_trans, dense_.shape(0), dense_.shape(1), matrix);
  #else
  if (kDLGPU == dense_.device_type()) {
    matrix = new T[matrix_size];
    Copy<T>(dense_.data<T>(), matrix_size, kDLGPU, kDLCPU, matrix);
  } else {
    matrix = (T* )dense_.data<T>();
  }
  #endif

  // Dense to sparse transformation.
  seqmm::SparseMatrixGetSize((T* )matrix, matrix_size, &nnz_);

  // n_rows_ = dense_.shape(0);
  // n_cols_ = dense_.shape(1);
    // FIXME: pretend dense_ has been transposed.
  n_rows_ = dense_.shape(1);
  n_cols_ = dense_.shape(0);

  // Allocate Host memory for sparse data.
  vals_ = new T[nnz_];
  col_indices_ = new int[nnz_];
  row_offsets_ = new int[n_rows_ + 1];
  row_indices_ = new int[n_rows_];
  seqmm::Dense2Sparse((T* )matrix, n_rows_, n_cols_, vals_, col_indices_, row_offsets_);
  std::iota(row_indices_, row_indices_ + n_rows_, 0);

  if (kDLGPU == dense_.device_type()) {
    dense_.Delete();
#if IF_TRANS == 1
    delete [] matrix_trans;
#else
    delete [] matrix;
#endif
  }

#if IF_TRANS == 1
  delete [] matrix;
#endif

  // Allocate GPU memory for sparse data.
  TT_ENFORCE_CUDA_SUCCESS(cudaMalloc(&d_vals_, nnz_ * sizeof(T)));
  TT_ENFORCE_CUDA_SUCCESS(cudaMalloc(&d_col_indices_, nnz_ * sizeof(int)));
  TT_ENFORCE_CUDA_SUCCESS(cudaMalloc(&d_row_offsets_, (n_rows_ + 1) * sizeof(int)));
  TT_ENFORCE_CUDA_SUCCESS(cudaMalloc(&d_row_indices_, n_rows_ * sizeof(int)));

  // Copy sparse data from Host to GPU.
  TT_ENFORCE_CUDA_SUCCESS(cudaMemcpy(d_vals_, vals_, nnz_ * sizeof(T),
                                     cudaMemcpyHostToDevice));
  TT_ENFORCE_CUDA_SUCCESS(cudaMemcpy(d_col_indices_, col_indices_,
                                     nnz_ * sizeof(int),
                                     cudaMemcpyHostToDevice));
  TT_ENFORCE_CUDA_SUCCESS(cudaMemcpy(d_row_offsets_, row_offsets_,
                                     (n_rows_ + 1) * sizeof(int),
                                     cudaMemcpyHostToDevice));
  TT_ENFORCE_CUDA_SUCCESS(cudaMemcpy(d_row_indices_, row_indices_,
                                     n_rows_ * sizeof(int),
                                     cudaMemcpyHostToDevice));

}

template<class T> void SparseTensorCsr<T>::Destroy() {
  delete [] vals_;
  delete [] col_indices_;
  delete [] row_offsets_;
  delete [] row_indices_;

  TT_ENFORCE_CUDA_SUCCESS(cudaFree(d_vals_));
  TT_ENFORCE_CUDA_SUCCESS(cudaFree(d_col_indices_));
  TT_ENFORCE_CUDA_SUCCESS(cudaFree(d_row_offsets_));
  TT_ENFORCE_CUDA_SUCCESS(cudaFree(d_row_indices_));
}

template void SparseTensorCsr<float>::Dense2Sparse();
template void SparseTensorCsr<float>::Destroy();

}
}
