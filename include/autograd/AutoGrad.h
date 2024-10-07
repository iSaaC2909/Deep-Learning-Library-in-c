#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "Tensor.h"

// Declare functions related to Autograd
void backward(Tensor& output);  // Compute gradients for output tensor
void reset_gradients(Tensor& tensor);  // Reset gradients in a tensor

#endif // AUTOGRAD_H
