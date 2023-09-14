#include <iostream>
#include <fstream>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "utils.h"

#include <cuda_fp16.h>

namespace seqmm {

template <typename T>
void GenDenseMatrix(T *dense_matrix, int row, int column) {
  // set seed for rand()
  srand((unsigned int)(time(NULL)));

  // generate dense matrix
  RandomInit(dense_matrix, row * column);
}


template <typename T>
void GenSparseMatrix(T* sparse_matrix, int row, int column, size_t sparsity) {
  RandomInit(sparse_matrix, row * column, sparsity);
}

template <typename T>
void PreProcessData(const char* data_file_path,
                    int k, int n,
                    T* dense_matrix) {
  std::ifstream in(std::string(data_file_path), std::ios::binary | std::ios::in);
  in.read((char *)dense_matrix, sizeof(T) * k * n);
  std::cout << in.gcount() << " bytes read" << std::endl;
  
  // close input file stream
  in.close();
}

template <typename T>
void WriteMatrixToFile(const T* sparse_matrix, const size_t size,
                       std::string& data_file_path) {
  std::ofstream out(data_file_path, std::ios::binary);
  out.write((char *)sparse_matrix, size);
  std::cout << out.tellp() << " bytes written" << std::endl;
  
  out.close();
}

template <typename T>
void RandomInit(T *data, int size, int sparsity) {
  const double diff = 1.000000;
  int count = 0;
  for (int i = 0; i < size; ++i) {
    if (sparsity) {
      if (rand() % 100 >= sparsity) {
        data[i] = rand() * 1.0 / RAND_MAX;
        count++;
      }
      else
        data[i] = 0.0;
    }
    else {
      data[i] = rand() / (1.0 * RAND_MAX);
    }
  }

  float real_sparsity = 1.0 - float(count) / size;
  if (abs(real_sparsity - float(sparsity)) > diff) {
    printf("Expected sparsity is %d%%.\nTrue sparsity is %.2f%%.\n", sparsity, real_sparsity * 100);
  }
}

template <typename T>
void PrintDiff(T *data1, T *data2,
               int width, int height, int iListLength, T fListTol) {
  printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
  int i,j,k;
  int error_count=0;

  for (j = 0; j < height; j++) {
    if (error_count < iListLength) {
      printf("\n  Row %d:\n", j);
    }

    for (i = 0; i < width; i++) {
      k = j * width + i;
      T fDiff = fabs(data1[k] - data2[k]);

      if (fDiff > fListTol) {
        if (error_count < iListLength) {
          printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j,
                 data1[k], data2[k], fDiff);
        }

        error_count++;
      }
    }
  }

  printf(" \n  Total Errors = %d\n", error_count);
}

template <typename T>
bool CompareL2fe(const T *reference, const T *data,
                 const unsigned int len, const T epsilon) {
  assert(epsilon >= 0);

  T error = 0.0;
  T ref = 0.0;

  for (unsigned int i = 0; i < len; ++i) {
    // std::cout << reference[i] << "," << data[i] << std::endl;
    T diff = reference[i] - data[i];
    error = error + diff * diff;
    ref = ref + reference[i] * reference[i];
  }

  T normRef = (T)sqrtf(ref);

  if (fabs(ref) < epsilon) {
    std::cerr << "ERROR, reference l2-norm is 0\n";
    return false;
  }

  T normError = (T)sqrtf(error);
  error = normError / normRef;
  bool result = error < epsilon;

  if (!result) {
    std::cerr << "ERROR, l2-norm error " << error << " is greater than epsilon "
              << epsilon << "\n";
  }

  return result;
}

template <typename T>
void SparseMatrixGetSize(T* matrix, int size, int* nnz) {
  int _nnz = 0;
  for (int i = 0; i < size; i++) {
    if (abs(matrix[i] - ZERO) > NNZ_EPSILON) {
      _nnz++;
    }
  }
  *nnz = _nnz;
}

template <typename T>
void Dense2Sparse(T* dense_matrix, int row, int column,
                  T* csr_values, 
                  int* csr_col_indices,
                  int* csr_row_offsets) {
  int _nnz = 0;
  csr_row_offsets[0] = 0;
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < column; j++ ) {
      if (abs(dense_matrix[i*column+j] - ZERO) > NNZ_EPSILON)  {
        csr_values[_nnz] = dense_matrix[i*column+j];
        csr_col_indices[_nnz] = j;
        _nnz++;
      }
    }
    csr_row_offsets[i+1] = _nnz;
  }
}

template <typename T>
void Transpose(const T* in, const int row, const int column, T* out) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < column; j++ ) {
      //out[j][i] = in[i][j];
      out[j * row + i] = in[i * column + j];
    }
  }
}

template <typename T>
void Prune(const T* in_matrix, const int m, const int n, const int threshold,
           T* out_matrix) {
  // FIXME: implement real prune functionality.
  memcpy(out_matrix, in_matrix, m * n * sizeof(T));
}

template <typename T>
void AnalyzeSparsePattern(const T* matrix, const int m, const int n) {
  int nnz = 0;
  int nnz_rows = 0;
  int nnz_cols = 0;

  bool *nnz_col_j = new bool[n];
  memset(nnz_col_j, 0, n * sizeof(bool));
  for (int i = 0; i < m; ++i) {
    bool nnz_row_i = false;
    for (int j = 0; j < n; ++j) {
      if (abs(matrix[i * n + j] - ZERO) > NNZ_EPSILON) {
        nnz++;
        nnz_row_i = true;
        nnz_col_j[j] = true;
      }
    }
    nnz_rows += nnz_row_i ? 1 : 0;
  }

  for (int j = 0; j < n; ++j) {
    nnz_cols += nnz_col_j[j];
  }

  std::cout << "Number of nonzero elements: " << nnz << std::endl;
  std::cout << "Number of nonzero rows: " << nnz_rows << std::endl;
  std::cout << "Number of nonzero cols: " << nnz_cols << std::endl;

  delete [] nnz_col_j;
}


template <typename T>
void PrintMatrix(T *matrix, int row, int col) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      printf("%.8f ", matrix[i*col+j]);
    }
    printf("\n");
  }
}

template <typename T>
void PrintVector(T *vector, int64_t size) {
  printf("[ ");
  for (int i = 0; i < size; i++) {
    printf("%.8f ", vector[i]);
  }
  printf("]\n");
}


template void GenDenseMatrix(half *dense_matrix, int row, int column);
template void GenDenseMatrix(float *dense_matrix, int row, int column);

template void GenSparseMatrix(half* sparse_matrix, int row, int column,
                              size_t sparsity);
template void GenSparseMatrix(float* sparse_matrix, int row, int column,
                              size_t sparsity);

template bool CompareL2fe(const half *reference, const half *data,
                          const unsigned int len, const half epsilon);
template bool CompareL2fe(const float *reference, const float *data,
                          const unsigned int len, const float epsilon);

template void SparseMatrixGetSize(half* matrix, int size, int* nnz);
template void SparseMatrixGetSize(float* matrix, int size, int* nnz);

template void Dense2Sparse(half* dense_matrix, int row, int column,
                           half* csr_values, 
                           int* csr_col_indices,
                           int* csr_row_offsets);
template void Dense2Sparse(float* dense_matrix, int row, int column,
                           float* csr_values, 
                           int* csr_col_indices,
                           int* csr_row_offsets);

template void Transpose(const half* in, const int row, const int column,
                        half* out);
template void Transpose(const float* in, const int row, const int column,
                        float* out);

template void PreProcessData(const char* data_file_path,
                             int k, int n,
                             half* dense_matrix);
template void PreProcessData(const char* data_file_path,
                             int k, int n,
                             float* dense_matrix);

template void WriteMatrixToFile(const half* sparse_matrix, const size_t size,
                                std::string& data_file_path);
template void WriteMatrixToFile(const float* sparse_matrix, const size_t size,
                                std::string& data_file_path);


template void Prune(const half* in_matrix, const int m, const int n,
                    const int threshold,
                    half* out_matrix);
template void Prune(const float* in_matrix, const int m, const int n,
                    const int threshold,
                    float* out_matrix);

template void AnalyzeSparsePattern(const half* matrix, const int m, const int n);
template void AnalyzeSparsePattern(const float* matrix, const int m, const int n);
}  // namespace seqmm
