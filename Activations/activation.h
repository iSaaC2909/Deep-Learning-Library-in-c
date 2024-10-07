#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "tensor.h"

Tensor relu(const Tensor& x);
Tensor sigmoid(const Tensor& x);

#endif // ACTIVATION_H
