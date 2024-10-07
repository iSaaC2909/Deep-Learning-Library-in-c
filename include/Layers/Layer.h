#ifndef LAYER_H
#define LAYER_H

#include "Tensor.h"

// Abstract base class for all layers
class Layer {
public:
    virtual ~Layer() = default;

    // Forward pass: computes the output tensor given an input tensor
    virtual Tensor forward(const Tensor& input) = 0;

    // Backward pass: computes gradients with respect to input and updates weights
    virtual Tensor backward(const Tensor& grad_output) = 0;

    // Get trainable parameters (weights and biases)
    virtual std::vector<Tensor*> get_parameters() = 0;

    // Get gradients of trainable parameters
    virtual std::vector<Tensor*> get_gradients() = 0;
};

#endif // LAYER_H
