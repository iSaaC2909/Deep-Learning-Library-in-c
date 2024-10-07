#include "Autograd.h"

// Implement backward propagation logic
void backward(Tensor& output) {
    // Initialize gradients if not done already
    if (output.grad().empty()) {
        output.grad().resize(output.size());
        output.grad().fill(1.0f);  // Gradient w.r.t itself is 1
    }

    // Propagate the gradient through the computational graph
    if (output.backward_fn) {
        output.backward_fn(output);  // Call the backward function for the operation
    }
}

void reset_gradients(Tensor& tensor) {
    if (tensor.requires_grad()) {
        tensor.grad().zero();  // Reset the gradients to zero
    }
}


class Tensor {
public:
    Tensor(const std::vector<int>& dims, bool requires_grad = false);
    
    // Getters for gradient and requires_grad
    Tensor& grad();
    bool requires_grad() const;
    
    // Methods for gradient handling
    void zero_grad();  // Reset gradients

private:
    std::vector<int> dims_;
    std::vector<float> data_;
    Tensor grad_;  // To store gradients
    bool requires_grad_;
};

class Tensor {
public:
    Tensor(const std::vector<int>& dims, bool requires_grad = false);

    // Accessor for the computational graph
    std::function<void(Tensor&)> backward_fn;

private:
    std::vector<int> dims_;
    std::vector<float> data_;
    Tensor grad_;
    bool requires_grad_;

    // Tracking parent tensors for the computational graph
    Tensor* parent1_;
    Tensor* parent2_;
};

Tensor Tensor::operator+(const Tensor& other) {
    Tensor result(dims_);
    result.data_ = data_ + other.data_;
    
    if (requires_grad_ || other.requires_grad_) {
        result.requires_grad_ = true;

        // Capture the current tensors (this and other) and define the backward function
        result.parent1_ = this;
        result.parent2_ = &other;

        result.backward_fn = [](Tensor& grad_output) {
            if (parent1_->requires_grad_)
                parent1_->grad_ += grad_output;
            if (parent2_->requires_grad_)
                parent2_->grad_ += grad_output;
        };
    }

    return result;
}
