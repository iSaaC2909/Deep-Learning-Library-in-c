#include "early_stopping.h"

EarlyStopping::EarlyStopping(int patience, float min_delta)
    : patience_(patience), min_delta_(min_delta), wait_(0), best_val_loss_(std::numeric_limits<float>::max()) {}

bool EarlyStopping::should_stop(float val_loss) {
    if (val_loss < best_val_loss_ - min_delta_) {
        best_val_loss_ = val_loss;
        wait_ = 0;  // Reset wait counter
    } else {
        ++wait_;  // No improvement, increase wait counter
    }
    
    return wait_ >= patience_;  // Stop if exceeded patience
}
