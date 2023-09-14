#ifndef TT_CORE_SPARSE_TENSOR_H_
#define TT_CORE_SPARSE_TENSOR_H_

#include "tensor.h"

#define IF_TRANS 1

namespace turbo_transformers {

namespace core {

class SparseTensor {
 public:
  SparseTensor(Tensor& dense) : dense_(std::move(dense)) {}
  
  virtual void Dense2Sparse() = 0;
  virtual void Destroy() = 0;

 protected:
  Tensor dense_;
};

template <class T>
class SparseTensorCsr : public SparseTensor {
 public:
  SparseTensorCsr(Tensor& dense) : SparseTensor(dense) {
    // TODO: transpose dense_
  }
  void Dense2Sparse();
  void Destroy();

  int n_rows() const { return n_rows_; }
  int n_cols() const { return n_cols_; }
  
  int nnz() const { return nnz_; }

  T* CpuData() const { return vals_; }
  int* CpuColIndices() const { return col_indices_; }
  int* CpuRowOffsets() const { return row_offsets_; }
  int* CpuRowIndices() const { return row_indices_; }

  T* GpuData() const { return d_vals_; }
  int* GpuColIndices() const { return d_col_indices_; }
  int* GpuRowOffsets() const { return d_row_offsets_; }
  int* GpuRowIndices() const { return d_row_indices_; }
  
 private:
  int n_rows_, n_cols_;
  
  int nnz_;
  T* vals_;
  int* col_indices_;
  int* row_offsets_;
  int* row_indices_;

  T* d_vals_;
  int* d_col_indices_;
  int* d_row_offsets_;
  int* d_row_indices_;
};

}  // namespace core
}  // namespace turbo_transformers

#endif  // TT_CORE_SPARSE_TENSOR_H_
