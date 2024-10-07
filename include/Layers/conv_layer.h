#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "tensor.h"
#include <functional>

Tensor conv2D(const Tensor& input, const std::vector<Tensor>& filters, 
              int stride = 1, int padding = 0, 
              std::function<Tensor(const Tensor&)> activation = nullptr);

#endif // CONV_LAYER_H
