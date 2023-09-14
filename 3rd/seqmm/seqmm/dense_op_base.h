#ifndef SEQMM_DENSE_OP_BASE_H_
#define SEQMM_DENSE_OP_BASE_H_

namespace seqmm {

template<class T>
class MatMul {
 public:
  virtual void Init(const float alpha, const float beta,
                    const bool transA, const bool transB,
                    const int m, const int n, const int k) = 0;
  
  virtual void Run(const T* d_mat_A, const T* d_mat_B, T* d_mat_C,
                   const int m, const int n, const int k) = 0;
  
  virtual void Clear() = 0;

 protected:
  int lda_, ldb_, ldc_;
};  // class MatMul

}  // namespace seqmm

#endif  // SEQMM_DENSE_OP_BASE_H_
