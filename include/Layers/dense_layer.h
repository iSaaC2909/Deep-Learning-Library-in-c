#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "tensor.h"
#include <functional>

Tensor denseLayer(const Tensor& input, const Tensor& weights, const Tensor& bias, 
                  std::function<Tensor(const Tensor&)> activation = nullptr);

#endif // DENSE_LAYER_H
