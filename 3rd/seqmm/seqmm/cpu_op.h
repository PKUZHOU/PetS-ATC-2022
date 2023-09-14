#ifndef SEQMM_CPU_OP_H_
#define SEQMM_CPU_OP_H_

#include "dense_op_base.h"
#include "utils.h"

namespace seqmm {

template<class T>
class CpuMM : public MatMul<T> {
 public:
  void Init(const float alpha, const float beta,
            const bool transA, const bool transB,
            const int m, const int n, const int k) {
    transA_ = transA;
    transB_ = transB;

    this->lda_ = transA ? m : k;
    this->ldb_ = transB ? k : n;
    this->ldc_ = n;
  }
  
  void Run(const T* A, const T* B, T* C,
           const int M, const int N, const int K) {
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j) {
        T sum = 0;
        T a, b;
        
        for (int k = 0; k < K; ++k) {
          a = A[i * K + k];
          if (transB_) {
            b = B[j * K + k];
          } else {
            b = B[k * N + j];
          }
          sum += a * b;
        }
        
        C[i * N + j] = sum;
      }
  }
  
  void Clear() {}

 private:
  bool transA_;
  bool transB_;
};  // class CpuMM

}  // namespace seqmm

#endif  // SEQMM_CPU_OP_H_
