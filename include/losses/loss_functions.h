#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include <vector>
#include "tensor.h"  // Assuming your Tensor class is defined elsewhere

class LossFunctions {
public:
    // Cross-Entropy Loss function for multi-class classification
    static float cross_entropy_loss(const Tensor& predictions, const Tensor& labels);

     // Mean Squared Error (MSE) function for regression tasks
     static float mean_squared_error(const Tensor& predictions, const Tensor& labels);

     // Hinge Loss function for binary classification
    static float hinge_loss(const Tensor& predictions, const Tensor& labels)
};

#endif // LOSS_FUNCTIONS_H
