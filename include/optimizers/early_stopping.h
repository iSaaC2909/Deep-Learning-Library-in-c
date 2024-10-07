#ifndef EARLY_STOPPING_H
#define EARLY_STOPPING_H

class EarlyStopping {
public:
    EarlyStopping(int patience, float min_delta = 0.0f);

    // Call after each epoch with the validation loss
    bool should_stop(float val_loss);

private:
    int patience_;        // Maximum epochs to wait for improvement
    float min_delta_;     // Minimum change in validation loss to qualify as improvement
    int wait_;            // Epochs waited so far
    float best_val_loss_; // Best validation loss recorded
};

#endif // EARLY_STOPPING_H
