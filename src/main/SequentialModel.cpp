class SequentialModel {
public:
    SequentialModel() = default;

    // Add layers to the neural network
    void add_layer(Layer* layer);

    // Forward pass through the entire network
    Tensor forward(const Tensor& input);

    // Backward pass: propagate the gradients backward
    void backward(const Tensor& loss_grad);

    // Update weights for all layers
    void update_weights(float learning_rate);

private:
    std::vector<Layer*> layers; // List of layers
};

void SequentialModel::add_layer(Layer* layer) {
    layers.push_back(layer);
}

Tensor SequentialModel::forward(const Tensor& input) {
    Tensor output = input;
    for (auto& layer : layers) {
        output = layer->forward(output);
    }
    return output;
}

void SequentialModel::backward(const Tensor& loss_grad) {
    Tensor grad = loss_grad;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        (*it)->backward(grad);
        grad = (*it)->input_cache; // Cache contains the propagated gradient
    }
}

void SequentialModel::update_weights(float learning_rate) {
    for (auto& layer : layers) {
        layer->update_weights(learning_rate);
    }