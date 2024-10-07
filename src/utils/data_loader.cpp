#include "data_loader.h"
#include <fstream>
#include <algorithm>
#include <random>

DataLoader::DataLoader(const std::string& data_path, int batch_size)
    : batch_size_(batch_size), current_index_(0) {
    load_data(data_path);  // Load the data from the file
    shuffle();  // Shuffle data initially
}

void DataLoader::load_data(const std::string& data_path) {
    // Dummy function for loading data. Extend with actual file loading logic
    // Assuming that data_path contains paths to Tensor files
    // Load data into data_ vector
}

std::vector<Tensor> DataLoader::next_batch() {
    std::vector<Tensor> batch;
    for (int i = 0; i < batch_size_ && current_index_ < data_.size(); ++i, ++current_index_) {
        batch.push_back(data_[current_index_]);
    }
    return batch;
}

void DataLoader::shuffle() {
    // Random shuffle for training data
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data_.begin(), data_.end(), g);
}

bool DataLoader::has_next() const {
    return current_index_ < data_.size();
}
