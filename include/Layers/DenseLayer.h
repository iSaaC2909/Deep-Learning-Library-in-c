#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "Layer.h"
#include <vector>
#include <random>

// Fully Connected (Dense) Layer
class DenseLayer : public Layer {
public:
    DenseLayer(int input_size, int output_size);

    // Forward pass
    Tensor forward(const Tensor& input) override;

    // Backward pass
    Tensor backward(const Tensor& grad_output) override;

    // Get trainable parameters
    std::vector<Tensor*> get_parameters() override;

    // Get gradients of trainable parameters
    std::vector<Tensor*> get_gradients() override;

private:
    Tensor weights_;       // Weight matrix
    Tensor biases_;        // Bias vector
    Tensor grad_weights_;  // Gradient of weights
    Tensor grad_biases_;   // Gradient of biases

    Tensor input_cache_;   // Cache input for use in backward pass
};

#endif // DENSE_LAYER_H
