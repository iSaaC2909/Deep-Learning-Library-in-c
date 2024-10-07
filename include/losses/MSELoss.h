#ifndef MSE_LOSS_H
#define MSE_LOSS_H

#include "LossFunction.h"

// Mean Squared Error Loss
class MSELoss : public LossFunction {
public:
    // Compute loss
    float compute(const Tensor& predictions, const Tensor& targets) override;

    // Compute gradient of loss w.r. to predictions
    Tensor gradient(const Tensor& predictions, const Tensor& targets) override;
};

#endif // MSE_LOSS_H
