#ifndef SEQMM_TYPES_H_
#define SEQMM_TYPES_H_

namespace seqmm {
  
enum SparseFormat {
  kFmtNone = -1,
  kFmtCSR = 0,
  kFmtCSC,
  kFmtBELL
};

}  // namespace seqmm

#endif  // SEQMM_TYPES_H_
