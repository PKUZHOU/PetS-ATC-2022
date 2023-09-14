#include "turbo_transformers/core/cuda_utils.h"
#include "loguru.hpp"

namespace turbo_transformers {
namespace core {

float GetGpuMemUsage() {
  float free_m, total_m, used_m;
  size_t free_t, total_t;
  
  cudaMemGetInfo(&free_t, &total_t);
  free_m = free_t / 1048576.0;  // MB
  total_m = total_t / 1048576.0;
  used_m = total_m - free_m;

  LOG_S(INFO) << "Mem free " << free_t << "... " << free_m << " MB" << std::endl
              << "Mem total " << total_t << "... " << total_m << " MB" << std::endl
              << "Mem used " << used_m << " MB" << std::endl;
  
  return used_m;
}

}  // namespace core
}  // namespace turbo_transformers

