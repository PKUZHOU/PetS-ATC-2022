cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(SEQMM CUDA CXX C)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)

include(cmake/cuda.cmake)

create_cuda_gencode_flags(CUDA_GENCODE_FLAGS ${CUDA_ARCHS})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_GENCODE_FLAGS}")


include_directories("${PROJECT_SOURCE_DIR}")

add_subdirectory(seqmm)
