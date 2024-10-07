#include "NeuralNetwork.h"
#include "DenseLayer.h"
#include "SGD.h"
#include "MSELoss.h"
#include <vector>
#include <iostream>

int main() {
    // Define network architecture
    NeuralNetwork network(new MSELoss(), new SGD(0.01f));

    // Add layers
    network.add_layer(new DenseLayer(2, 3)); // Input layer: 2 neurons, Output layer: 3 neurons
    network.add_layer(new DenseLayer(3, 1)); // Hidden layer: 3 neurons, Output layer: 1 neuron

    // Example input and target tensors
    Tensor input({2, 1}, {1.0f, 2.0f});      // Batch size: 1, Features: 2
    Tensor target({1, 1}, {1.0f});          // Batch size: 1, Output: 1

    // Training loop (simple example)
    for (int epoch = 0; epoch < 1000; ++epoch) {
        float loss = network.train_step(input, target);
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << " Loss: " << loss << std::endl;
        }
    }

    return 0;
}
