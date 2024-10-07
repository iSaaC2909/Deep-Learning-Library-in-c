#include "SGD.h"
#include "TensorExceptions.h"

SGD::SGD(float learning_rate) : learning_rate_(learning_rate) {}

void SGD::update(std::vector<Tensor*>& parameters, std::vector<Tensor*>& gradients) {
    if (parameters.size() != gradients.size()) {
        throw DimensionMismatchException("Parameters and gradients must have the same size.");
    }

    for (size_t i = 0; i < parameters.size(); ++i) {
        Tensor* param = parameters[i];
        Tensor* grad = gradients[i];

        if (param->data_.size() != grad->data_.size()) {
            throw DimensionMismatchException("Parameter and gradient tensor sizes do not match.");
        }

        // Update each parameter: param = param - learning_rate * grad
        for (size_t j = 0; j < param->data_.size(); ++j) {
            param->data_[j] -= learning_rate_ * grad->data_[j];
        }
    }
}
