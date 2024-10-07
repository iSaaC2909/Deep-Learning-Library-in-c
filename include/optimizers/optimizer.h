#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include "tensor.h"

class Optimizer {
public:
    virtual void update(Tensor& weights, const Tensor& gradients) = 0;
    virtual ~Optimizer() = default;
};

class SGD : public Optimizer {
public:
    explicit SGD(float learning_rate);
    void update(Tensor& weights, const Tensor& gradients) override;

private:
    float learning_rate_;
};

class Momentum : public Optimizer {
public:
    Momentum(float learning_rate, float momentum);
    void update(Tensor& weights, const Tensor& gradients) override;

private:
    float learning_rate_;
    float momentum_;
    Tensor velocity_;
};

class Adam : public Optimizer {
public:
    Adam(float learning_rate, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8);
    void update(Tensor& weights, const Tensor& gradients) override;

private:
    float learning_rate_;
    float beta1_;
    float beta2_;
    float epsilon_;
    Tensor m_;  // First moment estimate
    Tensor v_;  // Second moment estimate
    int t_;     // Time step
};

class RMSprop : public Optimizer {
public:
    RMSprop(float learning_rate, float decay_rate = 0.99f, float epsilon = 1e-8);
    void update(Tensor& weights, const Tensor& gradients) override;

private:
    float learning_rate_;
    float decay_rate_;
    float epsilon_;
    Tensor cache_;
};

#endif // OPTIMIZERS_H
