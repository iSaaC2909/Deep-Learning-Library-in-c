#include "roc_auc.h"
#include <algorithm>
#include <tuple>

float ROCAUC::compute(const std::vector<float>& true_labels, const std::vector<float>& predicted_probs) {
    // Create vector of pairs: (predicted probability, true label)
    std::vector<std::pair<float, float>> sorted_pairs;
    for (size_t i = 0; i < true_labels.size(); ++i) {
        sorted_pairs.emplace_back(predicted_probs[i], true_labels[i]);
    }

    // Sort pairs by predicted probability in descending order
    std::sort(sorted_pairs.begin(), sorted_pairs.end(), std::greater<std::pair<float, float>>());

    // Initialize variables
    float tp = 0.0f, fp = 0.0f, auc = 0.0f, prev_tp = 0.0f, prev_fp = 0.0f;
    float total_pos = std::count(true_labels.begin(), true_labels.end(), 1.0f);
    float total_neg = true_labels.size() - total_pos;

    // Compute AUC using the trapezoidal rule
    for (const auto& pair : sorted_pairs) {
        float pred = pair.first;
        float label = pair.second;

        if (label == 1.0f) {
            tp += 1.0f;
        } else {
            fp += 1.0f;
        }

        if (fp != prev_fp) {
            auc += ((tp - prev_tp) * (fp - prev_fp)) / total_neg;
            prev_tp = tp;
            prev_fp = fp;
        }
    }

    auc /= total_pos;  // Normalize by the number of positives
    return auc;
}
