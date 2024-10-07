#include "MSELoss.h"
#include "TensorExceptions.h"
#include <cmath>

// Compute Mean Squared Error Loss
float MSELoss::compute(const Tensor& predictions, const Tensor& targets) {
    if (predictions.num_dimensions() != 2 || targets.num_dimensions() != 2) {
        throw InvalidTensorOperationException("MSELoss requires 2D tensors for predictions and targets.");
    }
    if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
        throw DimensionMismatchException("Predictions and targets must have the same dimensions.");
    }

    float loss = 0.0f;
    for (int i = 0; i < predictions.data_.size(); ++i) {
        float diff = predictions.data_[i] - targets.data_[i];
        loss += diff * diff;
    }
    return loss / predictions.data_.size();
}

// Compute gradient of MSE Loss w.r. to predictions
Tensor MSELoss::gradient(const Tensor& predictions, const Tensor& targets) {
    if (predictions.num_dimensions() != 2 || targets.num_dimensions() != 2) {
        throw InvalidTensorOperationException("MSELoss requires 2D tensors for predictions and targets.");
    }
    if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
        throw DimensionMismatchException("Predictions and targets must have the same dimensions.");
    }

    Tensor grad(predictions.dims_);
    for (int i = 0; i < predictions.data_.size(); ++i) {
        grad.data_[i] = 2.0f * (predictions.data_[i] - targets.data_[i]) / predictions.data_.size();
    }
    return grad;
}
