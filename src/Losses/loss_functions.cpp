#include "loss_functions.h"
#include "tensor.h"
#include <cmath>   // For std::log
#include <stdexcept>
#include <algorithm>

// Cross-Entropy Loss Implementation
float LossFunctions::cross_entropy_loss(const Tensor& predictions, const Tensor& labels) {
    if (predictions.num_dimensions() != 2 || labels.num_dimensions() != 2) {
        throw std::invalid_argument("Predictions and labels must be 2-dimensional (batch size x number of classes).");
    }
    
    int batch_size = predictions.size(0);  // Number of samples in the batch
    int num_classes = predictions.size(1); // Number of classes

    float total_loss = 0.0;

    // Iterate through each sample in the batch
    for (int i = 0; i < batch_size; ++i) {
        // For each sample, calculate cross-entropy for each class
        for (int j = 0; j < num_classes; ++j) {
            float predicted_prob = predictions.at({i, j});
            float true_label = labels.at({i, j});
            
            if (true_label == 1) { // Only add loss for the true class
                total_loss += -std::log(predicted_prob + 1e-8);  // Avoid log(0)
            }
        }
    }

    // Return the average loss over the batch
    return total_loss / batch_size;
}

// Mean Squared Error (MSE) Implementation
float LossFunctions::mean_squared_error(const Tensor& predictions, const Tensor& labels) {
    if (predictions.num_dimensions() != labels.num_dimensions()) {
        throw std::invalid_argument("Predictions and labels must have the same number of dimensions.");
    }

    // Check that the dimensions of predictions and labels match
    for (int i = 0; i < predictions.num_dimensions(); ++i) {
        if (predictions.size(i) != labels.size(i)) {
            throw std::invalid_argument("Predictions and labels must have the same shape.");
        }
    }

    int total_size = predictions.total_size(); // Assuming you have a total_size method to get the number of elements
    float total_error = 0.0;

    // Iterate through all elements
    for (int i = 0; i < total_size; ++i) {
        float diff = predictions.at(i) - labels.at(i);
        total_error += diff * diff;
    }

    // Return the average squared error
    return total_error / total_size;
}
//Hinge Loss implementation
float LossFunctions::hinge_loss(const Tensor& predictions, const Tensor& labels) {
    if (predictions.num_dimensions() != labels.num_dimensions()) {
        throw std::invalid_argument("Predictions and labels must have the same number of dimensions.");
    }

    // Check that the dimensions of predictions and labels match
    for (int i = 0; i < predictions.num_dimensions(); ++i) {
        if (predictions.size(i) != labels.size(i)) {
            throw std::invalid_argument("Predictions and labels must have the same shape.");
        }
    }

    int total_size = predictions.total_size(); // Assuming you have a total_size method
    float total_loss = 0.0;

    // Iterate through all elements
    for (int i = 0; i < total_size; ++i) {
        float prediction = predictions.at(i);
        float label = labels.at(i);  // Label should be +1 or -1 for Hinge Loss
        
        // Calculate hinge loss for each prediction
        total_loss += std::max(0.0f, 1.0f - label * prediction);
    }

    // Return the average hinge loss
    return total_loss / total_size;
}