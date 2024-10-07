#ifndef SGD_H
#define SGD_H

#include "Optimizer.h"

// Stochastic Gradient Descent Optimizer
class SGD : public Optimizer {
public:
    SGD(float learning_rate);

    // Update parameters
    void update(std::vector<Tensor*>& parameters, std::vector<Tensor*>& gradients) override;

private:
    float learning_rate_;
};

#endif // SGD_H
