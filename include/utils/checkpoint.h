#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include "tensor.h"
#include <string>

class Checkpoint {
public:
    Checkpoint(const std::string& checkpoint_dir);
    
    // Save model weights to file
    void save(const Tensor& model_weights, int epoch);
    
    // Load model weights from file
    Tensor load(const std::string& file_path);

private:
    std::string checkpoint_dir_;  // Directory to store checkpoints
};

#endif // CHECKPOINT_H
