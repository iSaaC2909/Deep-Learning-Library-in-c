#ifndef METRICS_H
#define METRICS_H

#include <vector>

class Metrics {
public:
    // Calculate accuracy
    static float accuracy(const std::vector<int>& true_labels, const std::vector<int>& predicted_labels);

    // Calculate precision
    static float precision(const std::vector<int>& true_labels, const std::vector<int>& predicted_labels);

    // Calculate recall
    static float recall(const std::vector<int>& true_labels, const std::vector<int>& predicted_labels);

    // Calculate F1 score
    static float f1_score(const std::vector<int>& true_labels, const std::vector<int>& predicted_labels);
};

#endif // METRICS_H
