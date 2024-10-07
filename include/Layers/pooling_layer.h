#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include "tensor.h"

Tensor maxPooling2D(const Tensor& input, int pool_height, int pool_width, int stride = 2, int padding = 0);
Tensor avgPooling2D(const Tensor& input, int pool_height, int pool_width, int stride = 2, int padding = 0);

#endif // POOLING_LAYER_H
