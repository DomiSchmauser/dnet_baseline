#include "vision.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "non-maximum suppression");
  m.def("nms3d", &nms3d, "3d non-maximum suppression");
}

