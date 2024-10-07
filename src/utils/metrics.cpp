#include "metrics.h"
#include <algorithm>  // For std::count

float Metrics::accuracy(const std::vector<int>& true_labels, const std::vector<int>& predicted_labels) {
    int correct = 0;
    for (size_t i = 0; i < true_labels.size(); ++i) {
        if (true_labels[i] == predicted_labels[i]) {
            ++correct;
        }
    }
    return static_cast<float>(correct) / true_labels.size();
}

float Metrics::precision(const std::vector<int>& true_labels, const std::vector<int>& predicted_labels) {
    int true_positive = 0, false_positive = 0;
    for (size_t i = 0; i < true_labels.size(); ++i) {
        if (predicted_labels[i] == 1) {
            if (true_labels[i] == 1) {
                ++true_positive;
            } else {
                ++false_positive;
            }
        }
    }
    return static_cast<float>(true_positive) / (true_positive + false_positive);
}

float Metrics::recall(const std::vector<int>& true_labels, const std::vector<int>& predicted_labels) {
    int true_positive = 0, false_negative = 0;
    for (size_t i = 0; i < true_labels.size(); ++i) {
        if (true_labels[i] == 1) {
            if (predicted_labels[i] == 1) {
                ++true_positive;
            } else {
                ++false_negative;
            }
        }
    }
    return static_cast<float>(true_positive) / (true_positive + false_negative);
}

float Metrics::f1_score(const std::vector<int>& true_labels, const std::vector<int>& predicted_labels) {
    float prec = precision(true_labels, predicted_labels);
    float rec = recall(true_labels, predicted_labels);
    if (prec + rec == 0) return 0.0f;  // Avoid division by zero
    return 2.0f * (prec * rec) / (prec + rec);
}
