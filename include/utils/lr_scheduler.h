#ifndef LR_SCHEDULER_H
#define LR_SCHEDULER_H

class LRScheduler {
public:
    // Constructor with initial learning rate
    explicit LRScheduler(float initial_lr);

    // Update learning rate based on epoch or performance
    virtual void step(int epoch) = 0;

    // Get the current learning rate
    float get_lr() const;

protected:
    float learning_rate_;
};

// Step decay scheduler
class StepLRScheduler : public LRScheduler {
public:
    StepLRScheduler(float initial_lr, int step_size, float gamma);
    void step(int epoch) override;

private:
    int step_size_;  // Number of epochs before decay
    float gamma_;    // Decay factor
};

#endif // LR_SCHEDULER_H
