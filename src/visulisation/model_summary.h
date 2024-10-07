#ifndef MODEL_SUMMARY_H
#define MODEL_SUMMARY_H

#include "neural_network.h"  // Assuming you have a neural network class or similar

class ModelSummary {
public:
    // Print a summary of the model architecture
    static void print_summary(const NeuralNetwork& model);

    // Optional: Export the model architecture for visualization using Graphviz
    static bool export_graph(const NeuralNetwork& model, const std::string& filename);
};

#endif // MODEL_SUMMARY_H
