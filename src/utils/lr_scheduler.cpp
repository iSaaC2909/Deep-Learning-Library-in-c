#include "lr_scheduler.h"

LRScheduler::LRScheduler(float initial_lr) : learning_rate_(initital_lr) {}

float LRSchedurler::get_lr() const {
    return learning_rate_;
}

StepLRScheduler::StepLRScheduler(float initial_lr, int step_size, float gamma)
    : LRScheduler(initial_lr), step_size_(step_size), gamma_(gamma) {}

void StepLRScheduler::step(int epoch) {
    if (epoch % step_size_ == 0 && epoch != 0) {
        learning_rate_ *= gamma_; //Decay the learning rate
    }
}