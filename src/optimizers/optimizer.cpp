#include "optimizer.h"
#include <cmath>

// Constructor for SGD, initializing the learning rate
SGD::SGD(float learning_rate) : learning_rate_(learning_rate) {}

// Update function for SGD, adjusting weights based on gradients and learning rate
void SGD::update(Tensor& weights, const Tensor& gradients) {
    if (weights.num_dimensions() != gradients.num_dimensions()) {
        throw std::invalid_argument("Weights and gradients must have the same number of dimensions.");
    }

    for (int i = 0; i < weights.total_size(); ++i) {
        // Adjust each weight by subtracting the gradient scaled by the learning rate
        weights.at(i) -= learning_rate_ * gradients.at(i);
    }
}

// Constructor for Momentum, initializing learning rate and momentum factor
Momentum::Momentum(float learning_rate, float momentum)
    : learning_rate_(learning_rate), momentum_(momentum), velocity_({}) {}

// Update function for Momentum optimizer
void Momentum::update(Tensor& weights, const Tensor& gradients) {
    if (weights.num_dimensions() != gradients.num_dimensions()) {
        throw std::invalid_argument("Weights and gradients must have the same number of dimensions.");
    }

    // Initialize velocity to zeros if not already initialized
    if (velocity_.total_size() == 0) {
        velocity_ = Tensor(weights.dims());
        velocity_.fill(0.0f);  // Initialize velocity tensor to zero
    }

    // Update the velocity and weights
    for (int i = 0; i < weights.total_size(); ++i) {
        // velocity = momentum * velocity - learning_rate * gradient
        velocity_.at(i) = momentum_ * velocity_.at(i) - learning_rate_ * gradients.at(i);
        // Update the weights using velocity
        weights.at(i) += velocity_.at(i);
    }
}

// Constructor for Adam optimizer
Adam::Adam(float learning_rate, float beta1, float beta2, float epsilon)
    : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon),
      m_({}), v_({}), t_(0) {}

// Update function for Adam optimizer
void Adam::update(Tensor& weights, const Tensor& gradients) {
    if (weights.num_dimensions() != gradients.num_dimensions()) {
        throw std::invalid_argument("Weights and gradients must have the same number of dimensions.");
    }

    ++t_;  // Increment time step

    // Initialize m and v tensors if they are not already
    if (m_.total_size() == 0) {
        m_ = Tensor(weights.dims());
        v_ = Tensor(weights.dims());
        m_.fill(0.0f);
        v_.fill(0.0f);
    }

    float beta1_t = pow(beta1_, t_);
    float beta2_t = pow(beta2_, t_);

    for (int i = 0; i < weights.total_size(); ++i) {
        // Update biased first moment estimate (mean)
        m_.at(i) = beta1_ * m_.at(i) + (1.0f - beta1_) * gradients.at(i);
        // Update biased second raw moment estimate (variance)
        v_.at(i) = beta2_ * v_.at(i) + (1.0f - beta2_) * gradients.at(i) * gradients.at(i);

        // Compute bias-corrected first and second moment estimates
        float m_hat = m_.at(i) / (1.0f - beta1_t);
        float v_hat = v_.at(i) / (1.0f - beta2_t);

        // Update weights
        weights.at(i) -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
    }
}