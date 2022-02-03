#pragma once
#include <torch/extension.h>
#include <THC/THC.h>

at::Tensor nms_cpu(const at::Tensor &dets,
                   const at::Tensor &scores,
                   const float threshold);

at::Tensor nms3d_cpu(const at::Tensor &dets,
                     const at::Tensor &scores,
                     const float threshold);