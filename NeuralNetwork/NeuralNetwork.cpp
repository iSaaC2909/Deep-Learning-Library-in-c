#include "NeuralNetwork.h"
#include "TensorExceptions.h"

// Constructor
NeuralNetwork::NeuralNetwork(LossFunction* loss_function, Optimizer* optimizer)
    : loss_function_(loss_function), optimizer_(optimizer) {}

// Add a layer to the network
void NeuralNetwork::add_layer(Layer* layer) {
    layers_.push_back(layer);
}

// Forward pass through all layers
Tensor NeuralNetwork::forward(const Tensor& input) {
    Tensor current = input;
    outputs_.clear();
    outputs_.push_back(current); // Input to the first layer

    for (auto& layer : layers_) {
        current = layer->forward(current);
        outputs_.push_back(current);
    }

    return current;
}

// Backward pass through all layers
void NeuralNetwork::backward(const Tensor& loss_grad) {
    Tensor grad = loss_grad;

    // Iterate through layers in reverse order
    for (int i = layers_.size() - 1; i >= 0; --i) {
        grad = layers_[i]->backward(grad);
    }
}

// Update network parameters using the optimizer
void NeuralNetwork::update() {
    // Collect all parameters and their gradients
    std::vector<Tensor*> parameters;
    std::vector<Tensor*> gradients;

    for (auto& layer : layers_) {
        std::vector<Tensor*> layer_params = layer->get_parameters();
        std::vector<Tensor*> layer_grads = layer->get_gradients();

        parameters.insert(parameters.end(), layer_params.begin(), layer_params.end());
        gradients.insert(gradients.end(), layer_grads.begin(), layer_grads.end());
    }

    // Update parameters using the optimizer
    optimizer_->update(parameters, gradients);
}

// Train the network on a single batch
float NeuralNetwork::train_step(const Tensor& input, const Tensor& target) {
    // Forward pass
    Tensor predictions = forward(input);

    // Compute loss
    float loss = loss_function_->compute(predictions, target);

    // Compute gradient of loss w.r. to predictions
    Tensor loss_grad = loss_function_->gradient(predictions, target);

    // Backward pass
    backward(loss_grad);

    // Update parameters
    update();

    return loss;
}
