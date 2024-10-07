#include "Tensor.h"
#include <iostream>
#include "MSELoss.h"
#include "Tensor.cpp"


int main() {
    // Create two tensors, both requiring gradients
    Tensor a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, true);
    Tensor b({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f}, true);

    // Perform addition
    Tensor c = a + b;

    // Perform backward pass (compute gradients)
    c.backward();

    // Output the gradients of a and b
    std::cout << "Gradient of a: ";
    for (int i = 0; i < a.size(0); ++i) {
        std::cout << a.grad.at({i}) << " ";
    }
    std::cout << std::endl;

    std::cout << "Gradient of b: ";
    for (int i = 0; i < b.size(0); ++i) {
        std::cout << b.grad.at({i}) << " ";
    }
    std::cout << std::endl;

    return 0;
}

