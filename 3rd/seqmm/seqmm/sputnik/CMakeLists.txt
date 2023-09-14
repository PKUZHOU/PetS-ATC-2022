add_library(sputnik OBJECT
  cuda_spmm.cu
  index_format.cu
  #matrix_utils.cu
  )
target_link_libraries(sputnik PUBLIC cudart cuda)
