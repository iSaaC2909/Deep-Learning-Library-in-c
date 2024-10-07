#include "Tensor.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iostrean>
#include "rnn_layer.h"

// Save tensor to a binary file
void Tensor::save_binary(const std::string& filename) const {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open()) {
        throw TensorException("Failed to open file for binary saving: " + filename);
    }
    int num_dims = dims_.size();
    ofs.write(reinterpret_cast<const char*>(&num_dims), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(dims_.data()), sizeof(int) * num_dims);
    ofs.write(reinterpret_cast<const char*>(data_.data()), sizeof(float) * data_.size());
    ofs.close();
}

// Load tensor from a binary file
void Tensor::load_binary(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) {
        throw TensorException("Failed to open file for binary loading: " + filename);
    }
    int num_dims;
    ifs.read(reinterpret_cast<char*>(&num_dims), sizeof(int));
    dims_.resize(num_dims);
    ifs.read(reinterpret_cast<char*>(dims_.data()), sizeof(int) * num_dims);
    total_size_ = compute_total_size(dims_);
    data_.resize(total_size_);
    ifs.read(reinterpret_cast<char*>(data_.data()), sizeof(float) * total_size_);
    ifs.close();
}

// Save tensor to a text file (CSV format)
void Tensor::save_text(const std::string& filename) const {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        throw TensorException("Failed to open file for text saving: " + filename);
    }
    ofs << dims_.size();
    for (const auto& dim : dims_) {
        ofs << "," << dim;
    }
    ofs << "\n";
    for (size_t i = 0; i < data_.size(); ++i) {
        ofs << std::fixed << std::setprecision(6) << data_[i];
        if (i != data_.size() - 1) ofs << ",";
    }
    ofs << "\n";
    ofs.close();
}

// Load tensor from a text file (CSV format)
void Tensor::load_text(const std::string& filename) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        throw TensorException("Failed to open file for text loading: " + filename);
    }
    std::string line;
    std::getline(ifs, line);
    std::stringstream ss_dims(line);
    std::string token;
    std::vector<int> loaded_dims;
    std::getline(ss_dims, token, ',');
    int num_dims = std::stoi(token);
    for (int i = 0; i < num_dims; ++i) {
        std::getline(ss_dims, token, ',');
        loaded_dims.push_back(std::stoi(token));
    }
    dims_ = loaded_dims;
    total_size_ = compute_total_size(dims_);
    data_.resize(total_size_);
    std::getline(ifs, line);
    std::stringstream ss_data(line);
    for (int i = 0; i < total_size_; ++i) {
        std::getline(ss_data, token, ',');
        data_[i] = std::stof(token);
    }
    ifs.close();
}

class NeuralNetwork {
public:
    Tensor forward(const Tensor& input);
    void backward(const Tensor& gradients);
    void update_weights(float learning_rate);
};

void Tensor::check_bounds(const std::vector<int>& indices) const {
    if (indices.size() != dims_.size()) {
        throw IndexOutOfBoundsException("Number of indices does not match tensor dimensions.");
    }
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < 0 || indices[i] >= dims_[i]) {
            throw IndexOutOfBoundsException("Index " + std::to_string(i) + " is out of bounds.");
        }
    }
}

Tensor Tensor::operator/(const Tensor& other) const {
    check_same_shape(other);
    Tensor result(dims_);
    for (size_t i = 0; i < data_.size(); ++i) {
        if (other.data_[i] == 0.0f) {
            throw DivisionByZeroException("Attempted division by zero.");
        }
        result.data_[i] = data_[i] / other.data_[i];
    }
    return result;
}

void Tensor::check_same_shape(const Tensor& other) const {
    if (dims_ != other.dims_) {
        throw DimensionMismatchException("Tensor shapes do not match for the operation.");
    }
}

Tensor Tensor::reshape(const std::vector<int>& new_dims) const {
    int new_total_size = std::accumulate(new_dims.begin(), new_dims.end(), 1, std::multiplies<int>());
    if (new_total_size != total_size_) {
        throw InvalidTensorOperationException("Reshape would change total number of elements.");
    }
    Tensor reshaped(new_dims, data_); // Assuming data is reshaped correctly here.
    return reshaped;
}


float Tensor::sum() const {
    return std::accumulate(data_.begin(), data_.end(), 0.0f);
}

float Tensor::mean() const{
    if (total_size_ == 0) {
        throw std::runtime_error("Cannot compute mean of an empty tensor.");
    }
    return sum() / static_cast<float>(total_size_);
}

// Variance of all elements
float Tensor::variance() const {
    if (total_size_ == 0) {
        throw std::runtime_error("Cannot compute variance of an empty tensor.");
    }
    float m = mean();
    float var = 0.0f;
    for (const auto& val : data_) {
        var += (val - m) * (val - m);
    }
    return var / static_cast<float>(total_size_);
}

// Standard deviation of all elements
float Tensor::stddev() const {
    return std::sqrt(variance());
}

// Maximum element
float Tensor::max() const {
    if (data_.empty()) {
        throw std::runtime_error("Cannot find maximum of an empty tensor.");
    }
    return *std::max_element(data_.begin(), data_.end());
}

// Minimum element
float Tensor::min() const {
    if (data_.empty()) {
        throw std::runtime_error("Cannot find minimum of an empty tensor.");
    }
    return *std::min_element(data_.begin(), data_.end());
}

// (Optional) Sum along a specific dimension
std::vector<float> Tensor::sum(int dim) const {
    if (dim < 0 || dim >= dims_.size()) {
        throw std::out_of_range("Dimension index out of bounds.");
    }

    // Calculate the new dimensions after summing over 'dim'
    std::vector<int> new_dims = dims_;
    new_dims.erase(new_dims.begin() + dim);
    Tensor result(new_dims);
    std::fill(result.data_.begin(), result.data_.end(), 0.0f);

    // Iterate over all elements and accumulate sums
    for (int i = 0; i < total_size_; ++i) {
        // Convert linear index to multi-dimensional indices
        std::vector<int> indices(dims_.size());
        int idx = i;
        for (int d = dims_.size() - 1; d >= 0; --d) {
            indices[d] = idx % dims_[d];
            idx /= dims_[d];
        }

        // Sum over the specified dimension
        indices.erase(indices.begin() + dim);
        int new_idx = result.index(indices);
        result.data_[new_idx] += data_[i];
    }

    return result.data_;
}

// (Optional) Mean along a specific dimension
std::vector<float> Tensor::mean(int dim) const {
    std::vector<float> sums = sum(dim);
    int new_size = sums.size();
    std::vector<float> means(new_size, 0.0f);

    for (int i = 0; i < new_size; ++i) {
        // Calculate the number of elements summed over 'dim'
        int count = dims_[dim];
        means[i] = sums[i] / static_cast<float>(count);
    }

    return means;
}

// (Optional) Variance along a specific dimension
std::vector<float> Tensor::variance(int dim) const {
    if (total_size_ == 0) {
        throw std::runtime_error("Cannot compute variance of an empty tensor.");
    }

    // Calculate mean along the specified dimension
    std::vector<float> means = mean(dim);
    std::vector<float> variances(means.size(), 0.0f);

    // Iterate over all elements and accumulate squared differences
    for (int i = 0; i < total_size_; ++i) {
        // Convert linear index to multi-dimensional indices
        std::vector<int> indices(dims_.size());
        int idx = i;
        for (int d = dims_.size() - 1; d >= 0; --d) {
            indices[d] = idx % dims_[d];
            idx /= dims_[d];
        }

        // Calculate variance
        int dim_value = indices[dim];
        indices.erase(indices.begin() + dim);
        int new_idx = index(indices);
        variances[new_idx] += (data_[i] - means[new_idx]) * (data_[i] - means[new_idx]);
    }

    // Divide by the number of elements to get variance
    for (auto& var : variances) {
        var /= static_cast<float>(dims_[dim]);
    }

    return variances;
}

// (Optional) Standard deviation along a specific dimension
std::vector<float> Tensor::stddev(int dim) const {
    std::vector<float> vars = variance(dim);
    std::vector<float> stddevs(vars.size(), 0.0f);
    for (size_t i = 0; i < vars.size(); ++i) {
        stddevs[i] = std::sqrt(vars[i]);
    }
    return stddevs;
}

// (Optional) Maximum along a specific dimension
std::vector<float> Tensor::max(int dim) const {
    if (dims_.empty()) {
        throw std::runtime_error("Cannot compute max of an empty tensor.");
    }
    std::vector<float> max_vals;
    int new_size = 1;
    for (size_t i = 0; i < dims_.size(); ++i) {
        if (i != static_cast<size_t>(dim)) {
            new_size *= dims_[i];
        }
    }
    max_vals.reserve(new_size);
    max_vals.assign(new_size, -std::numeric_limits<float>::infinity());

    for (int i = 0; i < total_size_; ++i) {
        // Convert linear index to multi-dimensional indices
        std::vector<int> indices(dims_.size());
        int idx = i;
        for (int d = dims_.size() - 1; d >= 0; --d) {
            indices[d] = idx % dims_[d];
            idx /= dims_[d];
        }

        // Find the corresponding index in the reduced tensor
        std::vector<int> reduced_indices = indices;
        reduced_indices.erase(reduced_indices.begin() + dim);
        int new_idx = index(reduced_indices);

        // Update maximum
        if (data_[i] > max_vals[new_idx]) {
            max_vals[new_idx] = data_[i];
        }
    }

    return max_vals;
}

// (Optional) Minimum along a specific dimension
std::vector<float> Tensor::min(int dim) const {
    if (dims_.empty()) {
        throw std::runtime_error("Cannot compute min of an empty tensor.");
    }
    std::vector<float> min_vals;
    int new_size = 1;
    for (size_t i = 0; i < dims_.size(); ++i) {
        if (i != static_cast<size_t>(dim)) {
            new_size *= dims_[i];
        }
    }
    min_vals.reserve(new_size);
    min_vals.assign(new_size, std::numeric_limits<float>::infinity());

    for (int i = 0; i < total_size_; ++i) {
        // Convert linear index to multi-dimensional indices
        std::vector<int> indices(dims_.size());
        int idx = i;
        for (int d = dims_.size() - 1; d >= 0; --d) {
            indices[d] = idx % dims_[d];
            idx /= dims_[d];
        }

        // Find the corresponding index in the reduced tensor
        std::vector<int> reduced_indices = indices;
        reduced_indices.erase(reduced_indices.begin() + dim);
        int new_idx = index(reduced_indices);

        // Update minimum
        if (data_[i] < min_vals[new_idx]) {
            min_vals[new_idx] = data_[i];
        }
    }

    return min_vals;
}

// Constructor initializes the tensor with the given dimensions and default values
Tensor::Tensor(const std::vector<int>& dims) 
    : dims_(dims), total_size_(compute_total_size(dims)), data_(total_size_) {}

Tensor::Tensor(const std::vector<int>& dims, const std::vector<float>& data)
    : dims_(dims), total_size_(compute_total_size(dims)), data_(data) {
    if (data.size() != total_size_) {
        throw std::invalid_argument("Data size does not match tensor dimensions.");
    }
}

// Fill the tensor with a specific value
void Tensor::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

// Access elements with bounds checking
float& Tensor::at(const std::vector<int>& indices) {
    check_bounds(indices);
    return data_[index(indices)];
}

const float& Tensor::at(const std::vector<int>& indices) const {
    check_bounds(indices);
    return data_[index(indices)];
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (dims_ != other.dims_) {
        throw std::invalid_argument("Tensors must have the same dimensions for element-wise multiplication.");
    }
    Tensor result(dims_);
    for (int i = 0; i < total_size_; ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

// Element-wise division with another tensor
Tensor Tensor::operator/(const Tensor& other) const {
    if (dims_ != other.dims_) {
        throw std::invalid_argument("Tensors must have the same dimensions for element-wise division.");
    }
    Tensor result(dims_);
    for (int i = 0; i < total_size_; ++i) {
        if (other.data_[i] == 0) {
            throw std::invalid_argument("Division by zero encountered in tensor element-wise division.");
        }
        result.data_[i] = data_[i] / other.data_[i];
    }
    return result;
}

// In-place element-wise multiplication
Tensor& Tensor::operator*=(const Tensor& other) {
    if (dims_ != other.dims_) {
        throw std::invalid_argument("Tensors must have the same dimensions for in-place element-wise multiplication.");
    }
    for (int i = 0; i < total_size_; ++i) {
        data_[i] *= other.data_[i];
    }
    return *this;
}

// In-place element-wise division
Tensor& Tensor::operator/=(const Tensor& other) {
    if (dims_ != other.dims_) {
        throw std::invalid_argument("Tensors must have the same dimensions for in-place element-wise division.");
    }
    for (int i = 0; i < total_size_; ++i) {
        if (other.data_[i] == 0) {
            throw std::invalid_argument("Division by zero encountered in in-place tensor element-wise division.");
        }
        data_[i] /= other.data_[i];
    }
}

// Element-wise addition with another tensor
Tensor Tensor::operator+(const Tensor& other) const {
    if (dims_ != other.dims_) {
        throw std::invalid_argument("Tensors must have the same dimensions for addition.");
    }
    Tensor result(dims_);
    for (int i = 0; i < total_size_; ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

// Reshape the tensor to new dimensions
Tensor Tensor::reshape(const std::vector<int>& new_dims) const {
    int new_size = compute_total_size(new_dims);
    if (new_size != total_size_) {
        throw std::invalid_argument("New dimensions must match the total number of elements.");
    }
    Tensor result(new_dims);
    result.data_ = data_; // Copy data; no need to reallocate
    return result;
}

// Memory management: Release resources
void Tensor::release() {
    // This example uses std::vector which manages its own memory, but if
    // there were other dynamically allocated resources, you would release them here.
    data_.clear();
}

// Memory usage
size_t Tensor::memory_usage() const {
    // Calculate memory usage in bytes
    // sizeof(float) * number of elements
    return sizeof(float) * data_.size();
}

// Get number of dimensions
int Tensor::num_dimensions() const {
    return dims_.size();
}

// Get size along a dimension
int Tensor::size(int dim) const {
    if (dim < 0 || dim >= dims_.size()) {
        throw std::out_of_range("Dimension index out of bounds.");
    }
    return dims_[dim];
}

// Check if indices are within bounds
void Tensor::check_bounds(const std::vector<int>& indices) const {
    if (indices.size() != dims_.size()) {
        throw std::invalid_argument("Number of indices must match number of dimensions.");
    }
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < 0 || indices[i] >= dims_[i]) {
            throw std::out_of_range("Index out of bounds.");
        }
    }
}

// Compute linear index from multi-dimensional indices
int Tensor::index(const std::vector<int>& indices) const {
    int idx = 0;
    int stride = 1;
    for (int i = dims_.size() - 1; i >= 0; --i) {
        idx += indices[i] * stride;
        stride *= dims_[i];
    }
    return idx;
}

// Compute total size of the tensor
int Tensor::compute_total_size(const std::vector<int>& dims) const {
    return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
}

Tensor Tensor::transpose(const std::vector<int>& permutation) const {
    if (permutation.size() != dims_.size()) {
        throw std::invalid_argument("Permutation size must match number of dimensions.");
    }

    std::vector<int> new_dims(dims_.size());
    for (size_t i = 0; i < permutation.size(); ++i) {
        if (permutation[i] < 0 || permutation[i] >= dims_.size()) {
            throw std::out_of_range("Permutation index out of bounds.");
        }
        new_dims[i] = dims_[permutation[i]];
    }

    Tensor result(new_dims);
    // Populate result.data_ with the transposed data
    // This is a complex operation and will depend on the specific permutation
    // Implementation should handle indexing based on permutation

    return result;
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (dims_.size() != 2 || other.dims_.size() != 2) {
        throw std::invalid_argument("Matrix multiplication is only implemented for 2D tensors.");
    }
    if (dims_[1] != other.dims_[0]) {
        throw std::invalid_argument("Number of columns in the first matrix must equal the number of rows in the second matrix.");
    }

    int rows = dims_[0];
    int cols = other.dims_[1];
    Tensor result({rows, cols});

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float sum = 0;
            for (int k = 0; k < dims_[1]; ++k) {
                sum += at({i, k}) * other.at({k, j});
            }
            result.at({i, j}) = sum;
        }
    }

    return result;
}

// Example of implementing element-wise addition with broadcasting
Tensor Tensor::operator+(const Tensor& other) const {
    // Implement broadcasting logic here
    // For simplicity, assuming broadcasting is not implemented yet
    if (dims_ != other.dims_) {
        throw std::invalid_argument("Tensors must have the same dimensions for element-wise addition.");
    }
    Tensor result(dims_);
    for (int i = 0; i < total_size_; ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

// Get the size of a specific dimension dynamically
int Tensor::dynamic_size(int dim) const {
    if (dim < 0 || dim >= dims_.size()) {
        throw std::out_of_range("Dimension index out of bounds.");
    }
    return dims_[dim];
}

// Set new size for a specific dimension
void Tensor::set_size(int dim, int new_size) {
    if (dim < 0 || dim >= dims_.size()) {
        throw std::out_of_range("Dimension index out of bounds.");
    }
    if (new_size < 0) {
        throw std::invalid_argument("New size must be non-negative.");
    }

    // Adjust dimensions and total size
    dims_[dim] = new_size;
    total_size_ = compute_total_size(dims_);
    data_.resize(total_size_); // Resize data to fit new size
}

#include "Tensor.h"

Tensor::Tensor(const std::vector<int>& dims, bool requires_grad)
    : dims_(dims), requires_grad(requires_grad) {
    total_size_ = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
    data_.resize(total_size_);
}

Tensor::Tensor(const std::vector<int>& dims, const std::vector<float>& data, bool requires_grad)
    : dims_(dims), data_(data), requires_grad(requires_grad) {
    total_size_ = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
    if (data.size() != total_size_) {
        throw std::invalid_argument("Data size does not match tensor dimensions.");
    }
}

void Tensor::backward() {
    if (!requires_grad) {
        throw std::runtime_error("This tensor does not require gradients.");
    }

    // Initialize gradient to 1 if not already set
    if (grad.data_.empty()) {
        grad = Tensor(dims_, std::vector<float>(total_size_, 1.0f));
    }

    // Call the backward function if it exists (propagate the gradient)
    if (backward_fn) {
        backward_fn();
    }
}

void Tensor::zero_grad() {
    if (!grad.data_.empty()) {
        std::fill(grad.data_.begin(), grad.data_.end(), 0.0f);
    }
}

Tensor Tensor::operator+(const Tensor& other) const {
    if (dims_ != other.dims_) {
        throw std::invalid_argument("Tensor dimensions must match for addition.");
    }

    Tensor result = *this;  // Copy the tensor

    // Perform element-wise addition
    for (int i = 0; i < total_size_; ++i) {
        result.data_[i] = this->data_[i] + other.data_[i];
    }

    // If autograd is needed, set the backward function
    if (requires_grad || other.requires_grad) {
        result.requires_grad = true;

        // Define the backward function
        result.backward_fn = [this, &other]() {
            // Gradients for both tensors
            if (this->requires_grad) {
                for (int i = 0; i < total_size_; ++i) {
                    this->grad.data_[i] += result.grad.data_[i];
                }
            }
            if (other.requires_grad) {
                for (int i = 0; i < total_size_; ++i) {
                    other.grad.data_[i] += result.grad.data_[i];
                }
            }
        };
    }

    return result;
}

class Layer {
public:
    virtual Tensor forward(const Tensor& input) = 0;
    virtual void backward(const Tensor& grad_output) = 0;
    virtual void update_weights(float learning_rate) = 0;
};

class DenseLayer : public Layer {
public:
    DenseLayer(int input_size, int output_size, bool requires_grad = true);

    // Forward pass: y = xW + b
    Tensor forward(const Tensor& input) override;

    // Backward pass: propagate gradients to previous layers
    void backward(const Tensor& grad_output) override;

    // Update weights and biases using gradient descent
    void update_weights(float learning_rate) override;

private:
    Tensor weights; // Weight matrix
    Tensor bias;    // Bias term
    Tensor grad_weights; // Gradients for the weights
    Tensor grad_bias;    // Gradients for the bias
    Tensor input_cache;  // Cache the input for backward pass

    bool requires_grad;
};

DenseLayer::DenseLayer(int input_size, int output_size, bool requires_grad)
    : requires_grad(requires_grad) {
    // Initialize weights and bias
    weights = Tensor({input_size, output_size}, true);
    bias = Tensor({1, output_size}, true);
}

Tensor DenseLayer::forward(const Tensor& input) {
    // Cache input for backpropagation
    input_cache = input;

    // Compute output: y = xW + b
    return input.matmul(weights) + bias;
}

void DenseLayer::backward(const Tensor& grad_output) {
    if (requires_grad) {
        // Compute gradients for weights and bias
        grad_weights = input_cache.transpose().matmul(grad_output);
        grad_bias = grad_output.sum({0}); // Sum along batch dimension

        // Propagate the gradient to the input layer (for further backpropagation)
        Tensor grad_input = grad_output.matmul(weights.transpose());
        input_cache = grad_input; // Cache for further backward passes
    }
}

void DenseLayer::update_weights(float learning_rate) {
    if (requires_grad) {
        // Update weights and biases using gradient descent
        weights = weights - grad_weights * learning_rate;
        bias = bias - grad_bias * learning_rate;
    }
}

class MSELoss {
public:
    // Forward pass: compute the loss
    Tensor forward(const Tensor& prediction, const Tensor& target);

    // Backward pass: compute the gradient of the loss with respect to predictions
    Tensor backward(const Tensor& prediction, const Tensor& target);
};

Tensor MSELoss::forward(const Tensor& prediction, const Tensor& target) {
    Tensor diff = prediction - target;
    return diff * diff.mean(); // Mean squared error
}

Tensor MSELoss::backward(const Tensor& prediction, const Tensor& target) {
    Tensor diff = prediction - target;
    return 2 * diff / prediction.size(0); // Gradient of MSE
}

Tensor Tensor::operator+(const Tensor& other) {
    if (dims_ != other.dims_) {
        throw std::invalid_argument("Dimension mismatch for addition");
    }

    Tensor result(dims_, requires_grad_ || other.requires_grad_);

    // Perform element-wise addition
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }

    // If either tensor requires gradient, track the parents
    if (result.requires_grad_) {
        result.parent1_ = std::make_shared<Tensor>(*this);
        result.parent2_ = std::make_shared<Tensor>(other);

        // Define the backward function for this operation
        result.backward_fn = [this, other](Tensor& grad_output) {
            if (this->requires_grad_) {
                this->grad_.data_.resize(this->data_.size());
                for (size_t i = 0; i < this->grad_.data_.size(); ++i) {
                    this->grad_.data_[i] += grad_output.data_[i];
                }
            }
            if (other.requires_grad_) {
                other.grad_.data_.resize(other.data_.size());
                for (size_t i = 0; i < other.grad_.data_.size(); ++i) {
                    other.grad_.data_[i] += grad_output.data_[i];
                }
            }
        };
    }

    return result;
}

void Tensor::backward() {
    // Initialize gradient if this is the starting point (e.g., the loss tensor)
    if (grad_.data_.empty()) {
        grad_.data_.resize(data_.size(), 1.0f);  // Gradient w.r.t itself is 1
    }

    // Call the backward function if one is defined
    if (backward_fn) {
        backward_fn(*this);  // Execute the backward function for the operation
    }

    // Recursively propagate gradients through parent tensors
    if (parent1_) {
        parent1_->backward();  // Backpropagate to parent 1
    }
    if (parent2_) {
        parent2_->backward();  // Backpropagate to parent 2
    }
}

Tensor denseLayer(const Tensor& input, const Tensor& weights, const Tensor& bias, 
                  std::function<Tensor(const Tensor&)> activation = nullptr) {
    // Step 1: Matrix Multiplication (input * weights)
    Tensor output = matmul(input, weights);

    // Step 2: Add the bias to each output neuron
    for (int i = 0; i < output.size(0); ++i) {
        for (int j = 0; j < output.size(1); ++j) {
            output.at({i, j}) += bias.at({j});
        }
    }

    // Step 3: Apply activation function (if provided)
    if (activation != nullptr) {
        output = activation(output);
    }

    return output;
}

Tensor conv2D(const Tensor& input, const std::vector<Tensor>& filters, 
              int stride = 1, int padding = 0, 
              std::function<Tensor(const Tensor&)> activation = nullptr) {
    int batch_size = input.size(0);
    int input_height = input.size(1);
    int input_width = input.size(2);
    int input_depth = input.size(3);  // Number of input channels (e.g., RGB)

    int num_filters = filters.size();  // Number of output channels
    int filter_height = filters[0].size(0);
    int filter_width = filters[0].size(1);

    // Calculate output dimensions
    int output_height = (input_height - filter_height + 2 * padding) / stride + 1;
    int output_width = (input_width - filter_width + 2 * padding) / stride + 1;

    // Initialize output tensor (batch_size, output_height, output_width, num_filters)
    Tensor output({batch_size, output_height, output_width, num_filters});
    
    // Padding input if needed
    Tensor padded_input = pad(input, padding);

    // Convolution operation
    for (int b = 0; b < batch_size; ++b) {
        for (int f = 0; f < num_filters; ++f) {
            for (int y = 0; y < output_height; ++y) {
                for (int x = 0; x < output_width; ++x) {
                    float sum = 0.0f;
                    // Perform convolution for each filter
                    for (int ky = 0; ky < filter_height; ++ky) {
                        for (int kx = 0; kx < filter_width; ++kx) {
                            for (int c = 0; c < input_depth; ++c) {
                                // Corresponding input pixel (taking stride into account)
                                int input_y = y * stride + ky;
                                int input_x = x * stride + kx;

                                // Element-wise multiplication and accumulation
                                sum += padded_input.at({b, input_y, input_x, c}) * filters[f].at({ky, kx, c});
                            }
                        }
                    }
                    // Set the computed sum at the correct output position
                    output.at({b, y, x, f}) = sum;
                }
            }
        }
    }

    // Apply activation function (if provided)
    if (activation != nullptr) {
        output = activation(output);
    }

    return output;
}

Tensor pad(const Tensor& input, int padding) {
    if (padding == 0) {
        return input;  // No padding needed
    }

    int batch_size = input.size(0);
    int input_height = input.size(1);
    int input_width = input.size(2);
    int input_depth = input.size(3);

    // Create a new tensor with padding added around the input
    Tensor padded_input({batch_size, input_height + 2 * padding, input_width + 2 * padding, input_depth});

    // Copy the input into the center of the padded tensor
    for (int b = 0; b < batch_size; ++b) {
        for (int y = 0; y < input_height; ++y) {
            for (int x = 0; x < input_width; ++x) {
                for (int c = 0; c < input_depth; ++c) {
                    padded_input.at({b, y + padding, x + padding, c}) = input.at({b, y, x, c});
                }
            }
        }
    }

    return padded_input;
}

Tensor maxPooling2D(const Tensor& input, int pool_height, int pool_width, int stride = 2, int padding = 0) {
    int batch_size = input.size(0);
    int input_height = input.size(1);
    int input_width = input.size(2);
    int input_depth = input.size(3);

    // Calculate output dimensions
    int output_height = (input_height - pool_height + 2 * padding) / stride + 1;
    int output_width = (input_width - pool_width + 2 * padding) / stride + 1;

    // Initialize output tensor
    Tensor output({batch_size, output_height, output_width, input_depth});

    // Apply padding if needed
    Tensor padded_input = pad(input, padding);

    // Perform max pooling
    for (int b = 0; b < batch_size; ++b) {
        for (int d = 0; d < input_depth; ++d) {
            for (int y = 0; y < output_height; ++y) {
                for (int x = 0; x < output_width; ++x) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    // Pooling window
                    for (int py = 0; py < pool_height; ++py) {
                        for (int px = 0; px < pool_width; ++px) {
                            int input_y = y * stride + py;
                            int input_x = x * stride + px;
                            max_val = std::max(max_val, padded_input.at({b, input_y, input_x, d}));
                        }
                    }
                    output.at({b, y, x, d}) = max_val;
                }
            }
        }
    }

    return output;
}

Tensor avgPooling2D(const Tensor& input, int pool_height, int pool_width, int stride = 2, int padding = 0) {
    int batch_size = input.size(0);
    int input_height = input.size(1);
    int input_width = input.size(2);
    int input_depth = input.size(3);

    // Calculate output dimensions
    int output_height = (input_height - pool_height + 2 * padding) / stride + 1;
    int output_width = (input_width - pool_width + 2 * padding) / stride + 1;

    // Initialize output tensor
    Tensor output({batch_size, output_height, output_width, input_depth});

    // Apply padding if needed
    Tensor padded_input = pad(input, padding);

    // Perform average pooling
    for (int b = 0; b < batch_size; ++b) {
        for (int d = 0; d < input_depth; ++d) {
            for (int y = 0; y < output_height; ++y) {
                for (int x = 0; x < output_width; ++x) {
                    float sum = 0.0f;
                    // Pooling window
                    for (int py = 0; py < pool_height; ++py) {
                        for (int px = 0; px < pool_width; ++px) {
                            int input_y = y * stride + py;
                            int input_x = x * stride + px;
                            sum += padded_input.at({b, input_y, input_x, d});
                        }
                    }
                    output.at({b, y, x, d}) = sum / (pool_height * pool_width);
                }
            }
        }
    }

    return output;
}

class RNNLayer {
public:
    RNNLayer(int input_size, int hidden_size) 
        : input_size(input_size), hidden_size(hidden_size) {
        // Initialize weights and biases
        W_h = Tensor({input_size, hidden_size});   // Input to hidden
        U_h = Tensor({hidden_size, hidden_size});  // Hidden to hidden
        b_h = Tensor({hidden_size});               // Bias
    }

    Tensor forward(const std::vector<Tensor>& inputs) {
        // Initialize hidden state
        Tensor h = Tensor({hidden_size});
        h.fill(0.0);

        // Iterate over the input sequence
        for (const Tensor& x : inputs) {
            h = tanh(W_h.matmul(x) + U_h.matmul(h) + b_h);
        }

        return h; // Final hidden state as the output
    }

private:
    int input_size, hidden_size;
    Tensor W_h, U_h, b_h;
};

class LSTMLayer {
public:
    LSTMLayer(int input_size, int hidden_size) 
        : input_size(input_size), hidden_size(hidden_size) {
        // Initialize weights for gates
        W_f = Tensor({input_size, hidden_size});
        U_f = Tensor({hidden_size, hidden_size});
        b_f = Tensor({hidden_size});

        W_i = Tensor({input_size, hidden_size});
        U_i = Tensor({hidden_size, hidden_size});
        b_i = Tensor({hidden_size});

        W_o = Tensor({input_size, hidden_size});
        U_o = Tensor({hidden_size, hidden_size});
        b_o = Tensor({hidden_size});

        W_c = Tensor({input_size, hidden_size});
        U_c = Tensor({hidden_size, hidden_size});
        b_c = Tensor({hidden_size});
    }

    Tensor forward(const std::vector<Tensor>& inputs) {
        // Initialize hidden and cell state
        Tensor h = Tensor({hidden_size});
        Tensor C = Tensor({hidden_size});
        h.fill(0.0);
        C.fill(0.0);

        for (const Tensor& x : inputs) {
            // Forget gate
            Tensor f_t = sigmoid(W_f.matmul(x) + U_f.matmul(h) + b_f);
            // Input gate
            Tensor i_t = sigmoid(W_i.matmul(x) + U_i.matmul(h) + b_i);
            // Output gate
            Tensor o_t = sigmoid(W_o.matmul(x) + U_o.matmul(h) + b_o);
            // Candidate cell state
            Tensor C_tilde = tanh(W_c.matmul(x) + U_c.matmul(h) + b_c);
            // Update cell state
            C = f_t * C + i_t * C_tilde;
            // Update hidden state
            h = o_t * tanh(C);
        }

        return h;
    }

private:
    int input_size, hidden_size;
    Tensor W_f, U_f, b_f;
    Tensor W_i, U_i, b_i;
    Tensor W_o, U_o, b_o;
    Tensor W_c, U_c, b_c;
};

class GRULayer {
public:
    GRULayer(int input_size, int hidden_size) 
        : input_size(input_size), hidden_size(hidden_size) {
        // Initialize weights
        W_z = Tensor({input_size, hidden_size});
        U_z = Tensor({hidden_size, hidden_size});
        b_z = Tensor({hidden_size});

        W_r = Tensor({input_size, hidden_size});
        U_r = Tensor({hidden_size, hidden_size});
        b_r = Tensor({hidden_size});

        W_h = Tensor({input_size, hidden_size});
        U_h = Tensor({hidden_size, hidden_size});
        b_h = Tensor({hidden_size});
    }

    Tensor forward(const std::vector<Tensor>& inputs) {
        // Initialize hidden state
        Tensor h = Tensor({hidden_size});
        h.fill(0.0);

        for (const Tensor& x : inputs) {
            // Update gate
            Tensor z_t = sigmoid(W_z.matmul(x) + U_z.matmul(h) + b_z);
            // Reset gate
            Tensor r_t = sigmoid(W_r.matmul(x) + U_r.matmul(h) + b_r);
            // Candidate hidden state
            Tensor h_tilde = tanh(W_h.matmul(x) + U_h.matmul(r_t * h) + b_h);
            // Update hidden state
            h = (1 - z_t) * h + z_t * h_tilde;
        }

        return h;
    }

private:
    int input_size, hidden_size;
    Tensor W_z, U_z, b_z;
    Tensor W_r, U_r, b_r;
    Tensor W_h, U_h, b_h;
};

class BatchNormalization {
public:
    BatchNormalization(int num_features, float epsilon = 1e-5, float momentum = 0.9)
        : num_features(num_features), epsilon(epsilon), momentum(momentum) {
        gamma = Tensor({num_features});
        beta = Tensor({num_features});
        running_mean = Tensor({num_features});
        running_var = Tensor({num_features});
        gamma.fill(1.0);  // Initialize gamma to 1
        beta.fill(0.0);   // Initialize beta to 0
        running_mean.fill(0.0);
        running_var.fill(1.0);
    }

    Tensor forward(const Tensor& input, bool training = true) {
        if (training) {
            // Calculate batch mean and variance
            Tensor batch_mean = input.mean(0);  // Mean across the batch
            Tensor batch_var = input.var(0);    // Variance across the batch

            // Normalize the input
            Tensor normalized_input = (input - batch_mean) / (batch_var.sqrt() + epsilon);

            // Update running statistics
            running_mean = momentum * running_mean + (1 - momentum) * batch_mean;
            running_var = momentum * running_var + (1 - momentum) * batch_var;

            return gamma * normalized_input + beta;  // Apply scale and shift
        } else {
            // Use running mean and variance during inference
            Tensor normalized_input = (input - running_mean) / (running_var.sqrt() + epsilon);
            return gamma * normalized_input + beta;
        }
    }

private:
    int num_features;
    float epsilon, momentum;
    Tensor gamma, beta;
    Tensor running_mean, running_var;
};

class LayerNormalization {
public:
    LayerNormalization(int num_features, float epsilon = 1e-5) 
        : num_features(num_features), epsilon(epsilon) {
        gamma = Tensor({num_features});
        beta = Tensor({num_features});
        gamma.fill(1.0);  // Initialize gamma to 1
        beta.fill(0.0);   // Initialize beta to 0
    }

    Tensor forward(const Tensor& input) {
        // Compute mean and variance across the features
        Tensor mean = input.mean(1);  // Mean across features
        Tensor variance = input.var(1);  // Variance across features

        // Normalize input
        Tensor normalized_input = (input - mean) / (variance.sqrt() + epsilon);

        // Scale and shift
        return gamma * normalized_input + beta;
    }

private:
    int num_features;
    float epsilon;
    Tensor gamma, beta;
};
