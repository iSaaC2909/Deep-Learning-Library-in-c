#ifndef MODEL_IO_H
#define MODEL_IO_H

#include <string>
#include "neural_network.h"  // Assuming you have a neural network class or similar

class ModelIO {
public:
    // Save the model to a file
    static bool save(const NeuralNetwork& model, const std::string& filename);

    // Load the model from a file
    static bool load(NeuralNetwork& model, const std::string& filename);
};

#endif // MODEL_IO_H
