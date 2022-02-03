#pragma once
#include <torch/extension.h>


at::Tensor nms_cuda(const at::Tensor boxes, float nms_overlap_thresh);
at::Tensor nms3d_cuda(const at::Tensor boxes, float nms_overlap_thresh);
