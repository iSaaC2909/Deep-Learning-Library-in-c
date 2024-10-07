#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <string>
#include "tensor.h"  // Assuming Tensor is defined elsewhere

class DataLoader {
public:
    // Constructor to load data from a file or dataset
    explicit DataLoader(const std::string& data_path, int batch_size);

    // Fetch the next batch of data
    std::vector<Tensor> next_batch();

    // Shuffle the data
    void shuffle();

    // Check if there are batches remaining
    bool has_next() const;

private:
    std::vector<Tensor> data_;  // Store all data tensors
    int batch_size_;            // Size of each batch
    int current_index_;         // Index for fetching next batch

    void load_data(const std::string& data_path);  // Helper to load data
};

#endif // DATA_LOADER_H
