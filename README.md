Tensors
Tensors are the fundamental data structures in this library, used to represent multi-dimensional arrays of data. They can be thought of as a generalization of matrices to higher dimensions. Each tensor can hold data of various types (e.g., floats, integers) and can support a variety of operations including arithmetic, reshaping, slicing, and broadcasting.

Key Features:

Multi-dimensional Support: Tensors can have any number of dimensions (n-dimensional), making them suitable for a wide range of applications in deep learning.
Operations: Supports element-wise operations (addition, subtraction, multiplication, division), matrix multiplication, reshaping, and transposing.
Memory Management: The library efficiently manages memory, allowing for dynamic allocation and deallocation as needed, thus optimizing performance.


Autograd
Automatic differentiation is a key feature of the library that allows for efficient computation of gradients, which are essential for optimizing neural networks through backpropagation. The autograd system tracks operations performed on tensors to create a computational graph, enabling the calculation of derivatives automatically.

Key Features:

Gradient Tracking: Each tensor can store its gradient, which is automatically updated during the backward pass.
Backward Functionality: For each operation on tensors, a corresponding backward function is defined to calculate how gradients flow through the network.
Efficient Computation: The library minimizes redundant calculations, making the gradient computation fast and efficient.

Neural Network Layers
The library supports various neural network layers, enabling the construction of complex models. Each layer can be stacked to form deep learning architectures.

Types of Layers:

Dense (Fully Connected) Layers: Connect every neuron from the previous layer to every neuron in the current layer, useful for high-level feature learning.
Convolutional Layers: Primarily used for processing image data, applying convolutional filters to extract spatial hierarchies of features.
Pooling Layers: Reduce the spatial dimensions of feature maps, retaining essential features while reducing computational load (e.g., max pooling, average pooling).
Recurrent Layers: Used for sequential data, such as time series or natural language, allowing the network to maintain a memory of previous inputs (e.g., LSTM, GRU).
Normalization Layers: Standardize the inputs to a layer, helping to stabilize learning and improve convergence (e.g., Batch Normalization).

Loss Functions
Loss functions quantify how well a neural network's predictions match the true labels. They are critical for guiding the optimization process during training.

Common Loss Functions:

Cross-Entropy Loss: Commonly used for classification tasks, measuring the performance of a model whose output is a probability value between 0 and 1.
Mean Squared Error (MSE): Used for regression tasks, measuring the average squared difference between predicted and actual values.
Hinge Loss: Typically used for "maximum-margin" classification, particularly with Support Vector Machines (SVMs).

Optimizers
Optimizers adjust the parameters of the model during training to minimize the loss function. Various optimization algorithms can be implemented to achieve faster convergence and better performance.

Popular Optimizers:

Stochastic Gradient Descent (SGD): A simple yet effective optimization algorithm that updates parameters based on a small batch of data.
Adam: Combines the advantages of two other extensions of SGD, namely AdaGrad and RMSProp, to provide adaptive learning rates for each parameter.
RMSprop: An adaptive learning rate method that divides the learning rate by an exponentially decaying average of squared gradients.


## API Reference

### 1. Tensor Class

The `Tensor` class is the core building block of this library, representing n-dimensional arrays. It provides a variety of methods for tensor manipulation, mathematical operations, and gradient tracking.

#### Constructors

- **Tensor(const std::vector<int>& dims)**
  - **Description**: Initializes a tensor with the specified dimensions.
  - **Parameters**: 
    - `dims`: A vector of integers representing the size of each dimension.
  
- **Tensor(const std::vector<int>& dims, const std::vector<float>& data)**
  - **Description**: Initializes a tensor with the specified dimensions and initial data.
  - **Parameters**: 
    - `dims`: A vector of integers representing the size of each dimension.
    - `data`: A vector of floats containing initial data.

#### Methods

- **void fill(float value)**
  - **Description**: Fills the tensor with a specific value.
  - **Parameters**: 
    - `value`: The float value to fill the tensor with.

- **float& at(const std::vector<int>& indices)**
  - **Description**: Accesses an element of the tensor at the specified indices.
  - **Parameters**: 
    - `indices`: A vector of integers representing the indices for the desired element.
  - **Returns**: A reference to the element at the specified indices.

- **const float& at(const std::vector<int>& indices) const**
  - **Description**: Accesses an element of the tensor at the specified indices (const version).
  - **Parameters**: 
    - `indices`: A vector of integers representing the indices for the desired element.
  - **Returns**: A constant reference to the element at the specified indices.

- **Tensor operator+(const Tensor& other) const**
  - **Description**: Performs element-wise addition with another tensor.
  - **Parameters**: 
    - `other`: The tensor to add to the current tensor.
  - **Returns**: A new tensor resulting from the addition.

- **Tensor reshape(const std::vector<int>& new_dims) const**
  - **Description**: Reshapes the tensor to the specified dimensions.
  - **Parameters**: 
    - `new_dims`: A vector of integers representing the new dimensions.
  - **Returns**: A new tensor with the reshaped dimensions.

- **void release()**
  - **Description**: Releases the memory allocated for the tensor.
  
- **size_t memory_usage() const**
  - **Description**: Returns the amount of memory used by the tensor in bytes.

- **int num_dimensions() const**
  - **Description**: Returns the number of dimensions of the tensor.
  - **Returns**: An integer representing the number of dimensions.

- **int size(int dim) const**
  - **Description**: Returns the size of the specified dimension.
  - **Parameters**: 
    - `dim`: The dimension index to query.
  - **Returns**: An integer representing the size of the specified dimension.

#### Mathematical Operations

- **Tensor operator-(const Tensor& other) const**
  - **Description**: Performs element-wise subtraction with another tensor.
  - **Parameters**: 
    - `other`: The tensor to subtract from the current tensor.
  - **Returns**: A new tensor resulting from the subtraction.

- **Tensor operator*(const Tensor& other) const**
  - **Description**: Performs element-wise multiplication with another tensor.
  - **Parameters**: 
    - `other`: The tensor to multiply with the current tensor.
  - **Returns**: A new tensor resulting from the multiplication.

- **Tensor operator/(const Tensor& other) const**
  - **Description**: Performs element-wise division by another tensor.
  - **Parameters**: 
    - `other`: The tensor to divide by the current tensor.
  - **Returns**: A new tensor resulting from the division.

- **Tensor matmul(const Tensor& other) const**
  - **Description**: Performs matrix multiplication with another tensor.
  - **Parameters**: 
    - `other`: The tensor to multiply with the current tensor.
  - **Returns**: A new tensor resulting from the matrix multiplication.

- **Tensor transpose() const**
  - **Description**: Transposes the tensor.
  - **Returns**: A new tensor that is the transpose of the current tensor.

- **Tensor reshape(const std::vector<int>& new_dims) const**
  - **Description**: Reshapes the tensor to the specified dimensions.
  - **Parameters**: 
    - `new_dims`: A vector of integers representing the new dimensions.
  - **Returns**: A new tensor with the reshaped dimensions.

### 2. Autograd Class

The `Autograd` class is responsible for automatic differentiation and managing the computational graph for tensors.

#### Methods

- **void backward(const Tensor& grad)**
  - **Description**: Computes gradients of tensors in the computational graph using backpropagation.
  - **Parameters**: 
    - `grad`: The gradient of the loss with respect to the output tensor.

- **void zero_grad()**
  - **Description**: Resets the gradients of all tensors in the computational graph to zero.

### 3. Neural Network Layers

The library supports various neural network layers. Each layer can be instantiated and used in the forward pass of a neural network.

#### Dense Layer Class

- **DenseLayer(int output_units)**
  - **Description**: Initializes a dense layer with the specified number of output units.
  - **Parameters**: 
    - `output_units`: The number of output units for this layer.

- **Tensor forward(const Tensor& input)**
  - **Description**: Performs a forward pass through the dense layer.
  - **Parameters**: 
    - `input`: The input tensor to the layer.
  - **Returns**: The output tensor after applying the layer's transformation.

#### Convolutional Layer Class

- **ConvLayer(int num_filters, int kernel_size)**
  - **Description**: Initializes a convolutional layer with the specified number of filters and kernel size.
  - **Parameters**: 
    - `num_filters`: The number of filters for the convolution.
    - `kernel_size`: The size of the convolutional kernel.

- **Tensor forward(const Tensor& input)**
  - **Description**: Performs a forward pass through the convolutional layer.
  - **Parameters**: 
    - `input`: The input tensor to the layer (usually a 4D tensor for images).
  - **Returns**: The output tensor after applying the convolution.

#### Pooling Layer Class

- **PoolingLayer(int pool_size)**
  - **Description**: Initializes a pooling layer with the specified pooling size.
  - **Parameters**: 
    - `pool_size`: The size of the pooling window.

- **Tensor forward(const Tensor& input)**
  - **Description**: Performs a forward pass through the pooling layer.
  - **Parameters**: 
    - `input`: The input tensor to the layer.
  - **Returns**: The output tensor after applying the pooling operation.

#### Recurrent Layer Class

- **RNNLayer(int hidden_units)**
  - **Description**: Initializes a recurrent layer with the specified number of hidden units.
  - **Parameters**: 
    - `hidden_units`: The number of hidden units for this recurrent layer.

- **Tensor forward(const Tensor& input)**
  - **Description**: Performs a forward pass through the recurrent layer.
  - **Parameters**: 
    - `input`: The input tensor to the layer (usually a 3D tensor for sequences).
  - **Returns**: The output tensor after applying the recurrent transformation.

### 4. Loss Functions

This section details the loss functions available in the library, each of which is essential for training neural networks.

#### Cross-Entropy Loss

- **float cross_entropy(const Tensor& predictions, const Tensor& targets)**
  - **Description**: Computes the cross-entropy loss between predictions and true labels.
  - **Parameters**: 
    - `predictions`: The predicted probabilities (output of the model).
    - `targets`: The true labels (ground truth).
  - **Returns**: The computed cross-entropy loss.

#### Mean Squared Error (MSE)

- **float mean_squared_error(const Tensor& predictions, const Tensor& targets)**
  - **Description**: Computes the mean squared error between predictions and true labels.
  - **Parameters**: 
    - `predictions`: The predicted values.
    - `targets`: The true labels.
  - **Returns**: The computed mean squared error.

#### Hinge Loss

- **float hinge_loss(const Tensor& predictions, const Tensor& targets)**
  - **Description**: Computes the hinge loss for binary classification tasks.
  - **Parameters**: 
    - `predictions`: The predicted values (scores).
    - `targets`: The true labels (-1 or 1).
  - **Returns**: The computed hinge loss.

### 5. Optimizers

Optimizers are responsible for updating the model parameters based on the gradients computed during backpropagation.

#### Stochastic Gradient Descent (SGD)

- **SGD(std::vector<Tensor>& parameters, float learning_rate)**
  - **Description**: Initializes the SGD optimizer with specified parameters and learning rate.
  - **Parameters**: 
    - `parameters`: A vector of tensors representing model parameters to optimize

.
    - `learning_rate`: The learning rate for the optimization step.

- **void step()**
  - **Description**: Performs a single optimization step to update model parameters.

#### Adam Optimizer

- **Adam(std::vector<Tensor>& parameters, float learning_rate, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8)**
  - **Description**: Initializes the Adam optimizer with specified parameters and hyperparameters.
  - **Parameters**: 
    - `parameters`: A vector of tensors representing model parameters to optimize.
    - `learning_rate`: The learning rate for the optimization step.
    - `beta1`: Exponential decay rate for the first moment estimate (default: 0.9).
    - `beta2`: Exponential decay rate for the second moment estimate (default: 0.999).
    - `epsilon`: A small constant to prevent division by zero (default: 1e-8).

- **void step()**
  - **Description**: Performs a single optimization step to update model parameters using the Adam algorithm.

#### RMSprop Optimizer

- **RMSprop(std::vector<Tensor>& parameters, float learning_rate, float decay_rate = 0.9, float epsilon = 1e-8)**
  - **Description**: Initializes the RMSprop optimizer with specified parameters and hyperparameters.
  - **Parameters**: 
    - `parameters`: A vector of tensors representing model parameters to optimize.
    - `learning_rate`: The learning rate for the optimization step.
    - `decay_rate`: Decay rate for the moving average of squared gradients (default: 0.9).
    - `epsilon`: A small constant to prevent division by zero (default: 1e-8).

- **void step()**
  - **Description**: Performs a single optimization step to update model parameters using the RMSprop algorithm.

### 6. Model Management

This section outlines how to manage the model, including saving/loading model parameters and viewing model summaries.

#### Save/Load Models

- **void save(const std::string& filename)**
  - **Description**: Saves the model parameters to a specified file.
  - **Parameters**: 
    - `filename`: The name of the file where the model will be saved.

- **void load(const std::string& filename)**
  - **Description**: Loads the model parameters from a specified file.
  - **Parameters**: 
    - `filename`: The name of the file from which to load the model.

#### Model Summary

- **void summary() const**
  - **Description**: Prints a summary of the model architecture, including the layers, their output shapes, and total parameters.

### 7. Evaluation Metrics

The library provides various functions for evaluating the performance of models.

#### Accuracy

- **float accuracy(const Tensor& predictions, const Tensor& targets)**
  - **Description**: Computes the accuracy of predictions against true labels.
  - **Parameters**: 
    - `predictions`: The predicted labels.
    - `targets`: The true labels.
  - **Returns**: The accuracy as a float between 0 and 1.

#### Precision

- **float precision(const Tensor& predictions, const Tensor& targets)**
  - **Description**: Computes the precision of predictions against true labels.
  - **Parameters**: 
    - `predictions`: The predicted labels.
    - `targets`: The true labels.
  - **Returns**: The precision as a float between 0 and 1.

#### Recall

- **float recall(const Tensor& predictions, const Tensor& targets)**
  - **Description**: Computes the recall of predictions against true labels.
  - **Parameters**: 
    - `predictions`: The predicted labels.
    - `targets`: The true labels.
  - **Returns**: The recall as a float between 0 and 1.

#### F1 Score

- **float f1_score(const Tensor& predictions, const Tensor& targets)**
  - **Description**: Computes the F1 score, which combines precision and recall into a single metric.
  - **Parameters**: 
    - `predictions`: The predicted labels.
    - `targets`: The true labels.
  - **Returns**: The F1 score as a float.

#### AUC-ROC

- **float auc_roc(const Tensor& predictions, const Tensor& targets)**
  - **Description**: Computes the area under the ROC curve for binary classification tasks.
  - **Parameters**: 
    - `predictions`: The predicted probabilities.
    - `targets`: The true binary labels.
  - **Returns**: The AUC value as a float.

### 8. Utilities

#### Data Loading and Preprocessing

- **void load_data(const std::string& filepath)**
  - **Description**: Loads data from the specified file path and prepares it for training.
  - **Parameters**: 
    - `filepath`: The path to the data file.

- **Tensor preprocess(const Tensor& data)**
  - **Description**: Preprocesses the input data (e.g., normalization, scaling).
  - **Parameters**: 
    - `data`: The input tensor containing raw data.
  - **Returns**: A preprocessed tensor ready for training.

### 9. GPU Acceleration (CUDA)

- **void enable_cuda()**
  - **Description**: Enables GPU acceleration using CUDA for tensor operations.
  
- **bool is_cuda_enabled() const**
  - **Description**: Checks if CUDA is enabled for the current session.
  - **Returns**: A boolean indicating whether CUDA is enabled.

### 10. Interoperability with Other Libraries

The library supports interoperability with popular Python libraries like NumPy and Pandas.

- **Tensor from_numpy(const np::ndarray& array)**
  - **Description**: Converts a NumPy array to a Tensor object.
  - **Parameters**: 
    - `array`: A NumPy array to convert.
  - **Returns**: A Tensor object.

- **np::ndarray to_numpy() const**
  - **Description**: Converts a Tensor object back to a NumPy array.
  - **Returns**: A NumPy array.

---
Prerequisites
C++ compiler (e.g., g++)
CUDA toolkit (for GPU acceleration)
#include "tensor.h"
#include "gpu_tensor.h"
#include "autograd.h"
// other includes as needed


git clone <repository-url>
cd deep-learning-library
