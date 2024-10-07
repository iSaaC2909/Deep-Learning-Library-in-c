#ifndef TENSOR_H
#define TENSOR_H

#include "TensorExceptions.h"
#include <vector>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <functional>

class Tensor {
public:
    
    Tensor(const std::vector<int>& dims, bool requires_grad = false);
    Tensor(const std::vector<int>& dims, const std::vector<float>& data, bool requires_grad = false);

    // Set backward function for automatic differentiation
    std::function<void()> backward_fn;

    // Gradient tensor (stores gradients during backpropagation)
    Tensor grad;
    
    // Flag to indicate if the tensor requires gradients
    bool requires_grad;

    // Perform backward pass (calculate gradients)
    void backward();

    // Zero the gradients (before performing a new backward pass)
    void zero_grad();

    // Other methods like reshape, transpose, etc.
    void fill(float value);
    Tensor reshape(const std::vector<int>& new_dims) const;

    // Access elements with bounds checking
    float& at(const std::vector<int>& indices);
    const float& at(const std::vector<int>& indices) const;

    // Element-wise addition with another tensor (automatic differentiation supported)
    Tensor operator+(const Tensor& other) const;

    // Memory management functions
    void release();
    size_t memory_usage() const;

    // Get number of dimensions and size along each dimension
    int num_dimensions() const;
    int size(int dim) const;

    private:
    std::vector<int> dims_;
    std::vector<float> data_;
    int total_size_;

    // Helper methods
    void check_bounds(const std::vector<int>& indices) const;
    int index(const std::vector<int>& indices) const;
    int compute_total_size(const std::vector<int>& dims) const;
    // Serialization methods
    void save_binary(const std::string& filename) const;  // Save tensor to a binary file
    void load_binary(const std::string& filename);        // Load tensor from a binary file
    void save_text(const std::string& filename) const;    // Save tensor to a text file (e.g., CSV format)
    void load_text(const std::string& filename);          // Load tensor from a text file (e.g., CSV format)


    // Statistical functions
    float sum() const;                              // Sum of all elements
    float mean() const;                             // Mean of all elements
    float variance() const;                         // Variance of all elements
    float stddev() const;                           // Standard deviation of all elements
    float max() const;                              // Maximum element
    float min() const;                              // Minimum element

    // (Optional) Statistical functions along a specific dimension
    std::vector<float> sum(int dim) const;         // Sum along a specified dimension
    std::vector<float> mean(int dim) const;        // Mean along a specified dimension
    std::vector<float> variance(int dim) const;    // Variance along a specified dimension
    std::vector<float> stddev(int dim) const;      // Standard deviation along a specified dimension
    std::vector<float> max(int dim) const;         // Maximum along a specified dimension
    std::vector<float> min(int dim) const;         // Minimum along a specified dimension

    // Element-wise operations with broadcasting support
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    // In-place operations with broadcasting support
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);

    // Constructors
    Tensor(const std::vector<int>& dims);
    Tensor(const std::vector<int>& dims, const std::vector<float>& data);

    // Data Manipulation
    void fill(float value);

    // Access elements with bounds checking
    float& at(const std::vector<int>& indices);
    const float& at(const std::vector<int>& indices) const;

    // Element-wise operations
    Tensor operator+(const Tensor& other) const;  // Addition
    Tensor operator-(const Tensor& other) const;  // Subtraction
    Tensor operator*(const Tensor& other) const;  // Multiplication
    Tensor operator/(const Tensor& other) const;  // Division

    // In-place element-wise operations
    Tensor& operator*=(const Tensor& other);       // In-place Multiplication
    Tensor& operator/=(const Tensor& other);       // In-place Division

    // Matrix multiplication
    Tensor matmul(const Tensor& other) const;

    // Transpose
    Tensor transpose(const std::vector<int>& permutation) const;

    // Reshape
    Tensor reshape(const std::vector<int>& new_dims) const;

    // Memory management
    void release();
    size_t memory_usage() const;

    // Metadata
    int num_dimensions() const;
    int size(int dim) const;

    // Dynamic dimensions
    int dynamic_size(int dim) const;
    void set_size(int dim, int new_size);

    Tensor(const std::vector<int>& dims, bool requires_grad = false);
    Tensor(const std::vector<int>& dims, const std::vector<float>& data, bool requires_grad = false);

    // Gradient tracking
    Tensor& grad();
    bool requires_grad() const;
    void zero_grad();

    // Forward pass operations (addition, multiplication, etc.)
    Tensor operator+(const Tensor& other);
    Tensor operator*(const Tensor& other);

    // Backward pass: propagate gradients through the computational graph
    void backward();

    // Store the backward function (for autograd)
    std::function<void(Tensor&)> backward_fn;

private:
    std::vector<int> dims_;
    std::vector<float> data_;
    int total_size_;
    
    std::vector<int> dims_;
    std::vector<float> data_;
    Tensor grad_;  // To store the gradient
    bool requires_grad_;

    // Parent tensors for the computational graph
    std::shared_ptr<Tensor> parent1_;
    std::shared_ptr<Tensor> parent2_;

    // Helper functions
    int total_size() const;

    // Helper methods
    void check_bounds(const std::vector<int>& indices) const;
    int index(const std::vector<int>& indices) const;
    int compute_total_size(const std::vector<int>& dims) const;
};

#endif // TENSOR_H
