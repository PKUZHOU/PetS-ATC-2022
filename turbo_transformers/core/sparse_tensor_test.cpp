#include "turbo_transformers/core/tensor.h"
#include "turbo_transformers/core/sparse_tensor.h"

#include "catch2/catch.hpp"

namespace turbo_transformers {
namespace core {

TEST_CASE("SparseTensorCsrTest", "[Dense2Sparse]") {
  turbo_transformers::core::Tensor dense_tensor(
      turbo_transformers::core::NewDLPackTensorT<float>({3, 4}));
  float *buff = dense_tensor.mutableData<float>();
  for (int i = 0; i < 12; ++i) buff[i] = i * 0.1;
  
  for (int i = 0; i < 12; ++i)
    REQUIRE(fabs(dense_tensor.data<float>()[i] - i * 0.1) < 1e-6);

  turbo_transformers::core::SparseTensorCsr<float> sparse_tensor(dense_tensor);

  sparse_tensor.Dense2Sparse();

  sparse_tensor.Destroy();
}

}  // namespace core
}  // namespace turbo_transformers
