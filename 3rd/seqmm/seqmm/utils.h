#ifndef SEQMM_UTILS_H_
#define SEQMM_UTILS_H_

#include <stdio.h>
#include <string>
#include <cublas_v2.h>
#include <cusparse_v2.h>

#define NNZ_EPSILON (1e-10)
#define ZERO 0.00

#define CHECK_CUDA(err)                                                 \
  do {                                                                  \
    if (err != cudaSuccess) {                                           \
      fprintf(stderr, "Fatal error: %d (%s at %s:%d)\n",                \
              err, cudaGetErrorString(err),                             \
              __FILE__, __LINE__);                                      \
      fprintf(stderr, "*** FAILED - ABORTING\n");                       \
      exit(1);                                                          \
    }                                                                   \
  } while (0)

#define CHECK_CUBLAS(err)                                               \
  do {                                                                  \
    if (err != CUBLAS_STATUS_SUCCESS) {                                 \
      fprintf(stderr, "Fatal error: %d (%s at %s:%d)\n",                \
              err, "Cublas Error",                                      \
              __FILE__, __LINE__);                                      \
      fprintf(stderr, "*** FAILED - ABORTING\n");                       \
      exit(1);                                                          \
    }                                                                   \
  } while (0)                                                           \

#define CHECK_CUSPARSE(err)                                             \
  do {                                                                  \
    if (err != CUSPARSE_STATUS_SUCCESS) {                               \
      fprintf(stderr, "Fatal error: %d (%s at %s:%d)\n",                \
              err, cusparseGetErrorString(err),                         \
              __FILE__, __LINE__);                                      \
      fprintf(stderr, "*** FAILED - ABORTING\n");                       \
      exit(1);                                                          \
    }                                                                   \
  } while (0)                                                           \


#define CUDA_EVENT_START(op, name)                                      \
  cudaEvent_t start_##op, stop_##op;                                    \
  CHECK_CUDA(cudaEventCreate(&start_##op));                             \
  CHECK_CUDA(cudaEventCreate(&stop_##op));                              \
  CHECK_CUDA(cudaEventRecord(start_##op));                              \
  
#define CUDA_EVENT_STOP(op, name)                                       \
  CHECK_CUDA(cudaEventRecord(stop_##op));                               \
  CHECK_CUDA(cudaEventSynchronize(stop_##op));                          \
  float time_##op = 0.0f;                                               \
  CHECK_CUDA(cudaEventElapsedTime(&time_##op, start_##op, stop_##op));  \
  std::cout << name << " time: " << time_##op / iter                    \
            << " ms" << std::endl;                                      \


namespace seqmm {

struct GPUContext {
  cublasHandle_t cublas_handle;
  cusparseHandle_t cusparse_handle;
  cudaStream_t stream;
};


/**
 * Generate a dense matrix.
 * input
 * - row, col: dimensions of dense matrix.
 * output
 * - dense_matrix: A dense matrix of shape (row * column).
 */
template <typename T>
void GenDenseMatrix(T* dense_matrix, int row, int column);


/**
 * Generate a dense matrix.
 * input
 * - row, col: dimensions of dense matrix.
 * - sparsity: pencentage of zero elements in sparse_matrix.
 * output
 * - sparse_matrix: A sparse matrix of shape (row * column).
 */
template <typename T>
void GenSparseMatrix(T* A, int m, int k, size_t density);

/**
 * Read data from a file and store it in a matrix.
 * input:
 * - data_file_path: path of data file
 * output:
 * - k: row dimension of output matrix
 * - n: column dimension of output matrix
 * - dense_matrix: matrix with stored data
 */
template <typename T>
void PreProcessData(const char* data_file_path,
                    int k, int n,
                    T* dense_matrix);

template <typename T>
void WriteMatrixToFile(const T* sparse_matrix, const size_t size,
                       std::string& data_file_path);

template <typename T>
void Transpose(const T* in, const int row, const int column, T* out);

/**
 * Prune sparse matrix according to nnz in row/col (threshold).
 * input:
 * - in_matrix: the input matrix to be pruned.
 * - m, n: row and col of in_matrix.
 * - threshold: nnz.
 * output:
 * - out_matrix: the pruned matrix.
 */
template <typename T>
void Prune(const T* in_matrix, const int m, const int n, const int threshold,
           T* out_matrix);

/**
 * Analyze sparse matrix pattern on given matrix.
 * input:
 * - matrix: the input matrix to be analyzed.
 * - m, n: row and col of in_matrix.
 * output:
 */
template <typename T>
void AnalyzeSparsePattern(const T* matrix, const int m, const int n);

/**
 * Allocates a matrix with random float entries.
 */
template <typename T>
void RandomInit(T* data, int size, int sparsity = 0);

template <typename T>
void PrintDiff(T *data1, T *data2,
               int width, int height, int iListLength, T fListTol);

template <typename T>
bool CompareL2fe(const T *reference, const T *data,
                 const unsigned int len, const T epsilon);

template <typename T>
void SparseMatrixGetSize(T* matrix, int size, int* nnz);

template <typename T>
void Dense2Sparse(T* dense_matrix, int row, int column,
                  T* csr_values, 
                  int* csr_col_indices,
                  int* csr_row_offsets);

template <typename T>
void PrintMatrix(T *matrix, int row, int col);

template <typename T>
void PrintVector(T *vector, int64_t size);

}  // namespace seqmm

#endif  // SEQMM_UTILS_H_
