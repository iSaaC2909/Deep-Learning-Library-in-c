#include "checkpoint.h"
#include <fstream>
#include <stdexcept>

Checkpoint::Checkpoint(const std::string& checkpoint_dir)
    : checkpoint_dir_(checkpoint_dir) {}

void Checkpoint::save(const Tensor& model_weights, int epoch) {
    std::string file_path = checkpoint_dir_ + "/model_epoch_" + std::to_string(epoch) + ".bin";
    std::ofstream out(file_path, std::ios::binary);
    if (!out) throw std::runtime_error("Error opening file for writing");

    // Save weights (implement serialization in Tensor class)
    model_weights.save(out);
    out.close();
}

Tensor Checkpoint::load(const std::string& file_path) {
    std::ifstream in(file_path, std::ios::binary);
    if (!in) throw std::runtime_error("Error opening file for reading");

    Tensor loaded_weights;
    loaded_weights.load(in);  // Implement deserialization in Tensor class
    in.close();
    return loaded_weights;
}
