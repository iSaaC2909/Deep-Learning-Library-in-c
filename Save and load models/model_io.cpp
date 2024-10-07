#include "model_io.h"
#include <fstream>
#include <iostream>

// Save model parameters to a file
bool ModelIO::save(const NeuralNetwork& model, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for saving model." << std::endl;
        return false;
    }

    // Serialize model parameters (weights and biases)
    for (const auto& layer : model.get_layers()) {
        for (const auto& weight : layer.weights) {
            file.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
        }
        for (const auto& bias : layer.biases) {
            file.write(reinterpret_cast<const char*>(&bias), sizeof(bias));
        }
    }

    file.close();
    return true;
}

// Load model parameters from a file
bool ModelIO::load(NeuralNetwork& model, const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for loading model." << std::endl;
        return false;
    }

    // Deserialize model parameters (weights and biases)
    for (auto& layer : model.get_layers()) {
        for (auto& weight : layer.weights) {
            file.read(reinterpret_cast<char*>(&weight), sizeof(weight));
        }
        for (auto& bias : layer.biases) {
            file.read(reinterpret_cast<char*>(&bias), sizeof(bias));
        }
    }

    file.close();
    return true;
}
