#ifndef NORMALIZATION_LAYER_H
#define NORMALIZATION_LAYER_H

#include "tensor.h"

class BatchNormalization {
public:
    BatchNormalization(int num_features, float epsilon = 1e-5, float momentum = 0.9);
    Tensor forward(const Tensor& input, bool training = true);

private:
    int num_features;
    float epsilon, momentum;
    Tensor gamma, beta;
    Tensor running_mean, running_var;
};

class LayerNormalization {
public:
    LayerNormalization(int num_features, float epsilon = 1e-5);
    Tensor forward(const Tensor& input);

private:
    int num_features;
    float epsilon;
    Tensor gamma, beta;
};

#endif // NORMALIZATION_LAYER_H
