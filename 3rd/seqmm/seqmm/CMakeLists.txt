add_subdirectory(sputnik)

set(SEQMM_SRCS)

list(APPEND SEQMM_SRCS
  sputnik_op.cpp
  utils.cpp)

cuda_add_library(seqmm ${SEQMM_SRCS})
target_include_directories(seqmm PUBLIC
  ${PROJECT_SOURCE_DIR}/seqmm/
  )

set(SEQMM_LIBS)

list(APPEND SEQMM_LIBS
  sputnik
  )

target_link_libraries(seqmm ${SEQMM_LIBS})


