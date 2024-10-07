#include "model_summary.h"
#include <iostream>
#include <fstream>

void ModelSummary::print_summary(const NeuralNetwork& model) {
    std::cout << "Model Summary:" << std::endl;
    std::cout << "========================" << std::endl;

    int layer_index = 1;
    for (const auto& layer : model.get_layers()) {
        std::cout << "Layer " << layer_index++ << ": " << layer.type << std::endl;
        std::cout << "Input Shape: " << layer.input_shape << std::endl;
        std::cout << "Output Shape: " << layer.output_shape << std::endl;
        std::cout << "Number of Parameters: " << layer.get_num_parameters() << std::endl;
        std::cout << "------------------------" << std::endl;
    }
    std::cout << "Total Parameters: " << model.get_total_parameters() << std::endl;
}

bool ModelSummary::export_graph(const NeuralNetwork& model, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for exporting graph." << std::endl;
        return false;
    }

    file << "digraph G {" << std::endl;
    file << "  rankdir=LR;" << std::endl;

    int layer_index = 1;
    for (const auto& layer : model.get_layers()) {
        file << "  Layer" << layer_index++ << " [label=\"" << layer.type 
             << "\\nInput Shape: " << layer.input_shape
             << "\\nOutput Shape: " << layer.output_shape 
             << "\\nParams: " << layer.get_num_parameters() << "\"];" << std::endl;
    }

    // Optionally, add edges between layers
    for (size_t i = 1; i < model.get_layers().size(); ++i) {
        file << "  Layer" << i << " -> Layer" << i + 1 << ";" << std::endl;
    }

    file << "}" << std::endl;
    file.close();
    return true;
}
