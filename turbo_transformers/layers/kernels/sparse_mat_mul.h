#ifndef TT_LAYERS_KERNELS_SPARSE_MAT_MUL_H_
#define TT_LAYERS_KERNELS_SPARSE_MAT_MUL_H_

#include "turbo_transformers/core/tensor.h"
#include "turbo_transformers/core/sparse_tensor.h"

#include "seqmm/types.h"
// #include "seqmm/cusparse_op.h"
#include "seqmm/sputnik_op.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {

class SparseMatMul{
 public: 
  virtual void Init() = 0;

  /**
   * Method to run sparse matrix multiplication: C=A*B.
   *
   * @input
   * --{A, B}: dense matrix and sparse matrix.
   * --{a_trans, b_trans}: whether A and B are transposed.
   * --{alpha, beta}: same as kernels::MatMul.
   * --cuda_ctx_ptr: pointer of CUDADeviceContext.
   * --name (optional): profiler tag.
   * @output
   * --C: result matrix.
   */
  virtual void Run(const core::Tensor& A, bool a_trans,
                   core::SparseTensor* B, bool b_trans,
                   float alpha, core::Tensor* C, float beta,
                   core::CUDADeviceContext* cuda_ctx_ptr,
                   const std::string name) = 0;
  virtual void Run(const core::Tensor& A, bool a_trans,
                   core::SparseTensor* B, bool b_trans,
                   float alpha, core::Tensor* C, float beta,
                   const std::string name) = 0;
  virtual void Clear() = 0;

  //static std::shared_ptr<SparseMatMul> SparseMatMulFactory(seqmm::SparseFormat fmt);
};

class SparseMatMulCsr : public SparseMatMul {
 public:
  SparseMatMulCsr();
  ~SparseMatMulCsr();
  
  void Init();

  static std::shared_ptr<SparseMatMulCsr> GetInstance(int stream_id, seqmm::SparseFormat fmt) {
    // FIXME: GetInstance for the nth task only works when tasks call
    // me in sequential order.
    // FIXME: support other formats.
    while (instances.size() <= stream_id)
    {
      std::shared_ptr<SparseMatMulCsr> instance =
        std::make_shared<SparseMatMulCsr>();
      instance->Init();
      instances.push_back(instance);
    }
    
    return instances.at(stream_id);
  } 

  void Run(const core::Tensor& A, bool a_trans,
           core::SparseTensor* B, bool b_trans,
           float alpha, core::Tensor* C, float beta,
           core::CUDADeviceContext* cuda_ctx_ptr,
           const std::string name);
  void Run(const core::Tensor& A, bool a_trans,
           core::SparseTensor* B, bool b_trans,
           float alpha, core::Tensor* C, float beta,
           const std::string name);
  void Clear();

 private:
  // seqmm::CuSparseMM<float> cusparse_spmm_;
  seqmm::SputnikMM sputnik_spmm_;

  static std::vector<std::shared_ptr<SparseMatMulCsr>> instances;
};

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers

#endif // TT_LAYERS_KERNELS_SPARSE_MAT_MUL_H_
