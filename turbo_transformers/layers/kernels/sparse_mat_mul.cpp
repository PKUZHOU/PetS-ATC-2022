
#include "sparse_mat_mul.h"
#include "turbo_transformers/core/cuda_device_context.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {

/*
std::shared_ptr<SparseMatMul>
SparseMatMul::SparseMatMulFactory(seqmm::SparseFormat fmt) {
  if (seqmm::SparseFormat::kFmtCSR == fmt) {
    return std::make_shared<SparseMatMulCsr>();
  } else {
    return nullptr;
  }
}
*/

SparseMatMulCsr::SparseMatMulCsr() {
  sputnik_spmm_.Init();
}

void SparseMatMulCsr::Init() {
  // cusparse_spmm_.Init();
  // sputnik_spmm_.Init();
}

void SparseMatMulCsr::Run(const core::Tensor& A, bool a_trans,
                          core::SparseTensor* B, bool b_trans,
                          float alpha, core::Tensor* C, float beta,
                          core::CUDADeviceContext* cuda_ctx_ptr,
                          const std::string name) {
  core::SparseTensorCsr<float>* csr_B = dynamic_cast<core::SparseTensorCsr<float>* >(B);
  
  if (csr_B) {
    int64_t m = A.shape(0) * A.shape(1);  // batch_size*seq_len
    // int64_t k = csr_B->n_rows();
    // int64_t n = csr_B->n_cols();
    int64_t n = csr_B->n_rows();
    int64_t k = csr_B->n_cols();
    int64_t nnz = csr_B->nnz();
    
    auto& cuda_ctx = (nullptr == cuda_ctx_ptr) ?
        ::turbo_transformers::core::CUDADeviceContext::GetInstance() : *cuda_ctx_ptr;

    /*
    cusparse_spmm_.Run(A.data<float>(), csr_B->GpuData(),
                       csr_B->GpuColIndices(), csr_B->GpuRowOffsets(),
                       nnz, m, n, k, false, false, 1.0, 0.0,
                       C->mutableData<float>(), cuda_ctx.cusparse_handle());
    */

#if IF_TRANS == 1
    sputnik_spmm_.Run(A.data<float>(), csr_B->GpuData(),
                      csr_B->GpuColIndices(), csr_B->GpuRowOffsets(),
                      csr_B->GpuRowIndices(),
                      nnz, m, n, k,
                      C->mutableData<float>(),
                      true,
                      cuda_ctx.stream());
#else
    sputnik_spmm_.Run(A.data<float>(), csr_B->GpuData(),
                      csr_B->GpuColIndices(), csr_B->GpuRowOffsets(),
                      csr_B->GpuRowIndices(),
                      nnz, m, n, k,
                      C->mutableData<float>(),
                      false,
                      cuda_ctx.stream());
#endif
  } else {
    std::cout << "Make sure B contains SparseTensorCsr object." << std::endl;
    exit(1);
  }
}

void SparseMatMulCsr::Run(const core::Tensor& A, bool a_trans,
                          core::SparseTensor* B, bool b_trans,
                          float alpha, core::Tensor* C, float beta,
                          const std::string name) {
  Run(A, a_trans, B, b_trans, alpha, C, beta, nullptr, name);
}

void SparseMatMulCsr::Clear() {
  // cusparse_spmm_.Clear();
  // sputnik_spmm_.Clear();
}

SparseMatMulCsr::~SparseMatMulCsr() {
  sputnik_spmm_.Clear();
}

std::vector<std::shared_ptr<SparseMatMulCsr>> SparseMatMulCsr::instances;

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_t
