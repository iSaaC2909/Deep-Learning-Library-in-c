#ifndef ROC_AUC_H
#define ROC_AUC_H

#include <vector>

class ROCAUC {
public:
    // Calculate AUC-ROC score
    static float compute(const std::vector<float>& true_labels, const std::vector<float>& predicted_probs);
};

#endif // ROC_AUC_H
