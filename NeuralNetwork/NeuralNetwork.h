#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Layer.h"
#include "Optimizer.h"
#include "LossFunction.h"
#include "Tensor.h"
#include <vector>

// Neural Network Model
class NeuralNetwork {
public:
    NeuralNetwork(LossFunction* loss_function, Optimizer* optimizer);

    // Add a layer to the network
    void add_layer(Layer* layer);

    // Forward pass through all layers
    Tensor forward(const Tensor& input);

    // Backward pass through all layers
    void backward(const Tensor& loss_grad);

    // Update network parameters using the optimizer
    void update();

    // Train the network on a single batch
    float train_step(const Tensor& input, const Tensor& target);

private:
    std::vector<Layer*> layers_;
    LossFunction* loss_function_;
    Optimizer* optimizer_;

    // Cache for storing outputs during forward pass
    std::vector<Tensor> outputs_;

    // Cache for storing gradients during backward pass
    std::vector<Tensor> gradients_;
};

#endif // NEURAL_NETWORK_H
