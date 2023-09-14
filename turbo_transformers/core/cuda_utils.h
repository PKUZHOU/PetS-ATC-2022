#ifndef TT_CORE_CUDA_UTILS_H_
#define TT_CORE_CUDA_UTILS_H_

#include <cuda_runtime.h>
#include <nvToolsExt.h>

//#define USE_NVTX

#ifdef USE_NVTX

const uint32_t colors[] = {0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff,
                           0xff00ffff, 0xffff0000, 0xffffffff};
const int tracer_num_colors = sizeof(colors) / sizeof(uint32_t);

#define PUSH_RANGE(name, cid) {                        \
    int color_id = cid;                                \
    color_id = color_id % tracer_num_colors;           \
    nvtxEventAttributes_t eventAttrib = {0};           \
    eventAttrib.version = NVTX_VERSION;                \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;  \
    eventAttrib.colorType = NVTX_COLOR_ARGB;           \
    eventAttrib.color = colors[color_id];              \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name;                  \
    nvtxRangePushEx(&eventAttrib);                     \
}

#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name, cid)
#define POP_RANGE
#endif

#ifdef USE_NVTX
class Tracer {
  nvtxRangeId_t id;

 public:
  Tracer(const char* name, int the_color_id) {
    int color_id = the_color_id;
    color_id = color_id % tracer_num_colors;
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = colors[color_id];
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = name;
    id = nvtxRangeStartEx(&eventAttrib);
    //    PUSH_RANGE(name, the_color_id);
  }
  ~Tracer() {
    nvtxRangeEnd(id);
    //    POP_RANGE;
  }
};
#define RANGE(name, color) Tracer __tracer(name, color);
#else
#define RANGE(name, color)
#endif

namespace turbo_transformers {
namespace core {

float GetGpuMemUsage();

}  // namespace core
}  // namespace turbo_transformers


#endif  // TT_CORE_CUDA_UTILS_H_
