#include "DenseLayer.h"
#include "TensorExceptions.h"

// Utility function to initialize weights with small random values
static float random_weight() {
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    return dist(gen);
}

DenseLayer::DenseLayer(int input_size, int output_size)
    : weights_({input_size, output_size}, std::vector<float>(input_size * output_size, 0.0f)),
      biases_({1, output_size}, std::vector<float>(output_size, 0.0f)),
      grad_weights_({input_size, output_size}, std::vector<float>(input_size * output_size, 0.0f)),
      grad_biases_({1, output_size}, std::vector<float>(output_size, 0.0f)) {

    // Initialize weights with small random values
    for (int i = 0; i < weights_.data_.size(); ++i) {
        weights_.data_[i] = random_weight();
    }

    // Initialize biases to zero
    biases_.fill(0.0f);
}

Tensor DenseLayer::forward(const Tensor& input) {
    input_cache_ = input; // Cache input for backward pass

    // Compute weights^T * input + biases
    Tensor output = input.matmul(weights_).reshape(input.rows(), weights_.cols());
    
    // Add biases (broadcasted)
    for (int i = 0; i < output.rows(); ++i) {
        for (int j = 0; j < output.cols(); ++j) {
            output.at({i, j}) += biases_.at({0, j});
        }
    }

    return output;
}

Tensor DenseLayer::backward(const Tensor& grad_output) {
    // Compute gradients w.r. to weights and biases
    // grad_weights = input^T * grad_output
    grad_weights_ = input_cache_.transpose({1, 0}).matmul(grad_output);

    // grad_biases = sum of grad_output over the batch dimension
    grad_biases_ = Tensor({1, grad_output.cols()});
    grad_biases_.fill(0.0f);
    for (int i = 0; i < grad_output.rows(); ++i) {
        for (int j = 0; j < grad_output.cols(); ++j) {
            grad_biases_.at({0, j}) += grad_output.at({i, j});
        }
    }

    // Compute gradient w.r. to input for previous layer
    Tensor grad_input = grad_output.matmul(weights_.transpose({1, 0}));

    return grad_input;
}

std::vector<Tensor*> DenseLayer::get_parameters() {
    return { &weights_, &biases_ };
}

std::vector<Tensor*> DenseLayer::get_gradients() {
    return { &grad_weights_, &grad_biases_ };
}
