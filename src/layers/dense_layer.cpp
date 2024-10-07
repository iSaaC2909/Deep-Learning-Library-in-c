#include "dense_layer.h"
#include "activation.h"  // If needed for activations

Tensor denseLayer(const Tensor& input, const Tensor& weights, const Tensor& bias, 
                  std::function<Tensor(const Tensor&)> activation) {
    // Implementation of dense layer (fully connected layer)
    Tensor output = matmul(input, weights);  // Matrix multiplication
    // Add bias and apply activation (code omitted for brevity)
    return output;
}
