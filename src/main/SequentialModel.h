int other() {
    // Create a model with one hidden layer
    SequentialModel model;
    model.add_layer(new DenseLayer(3, 5));  // Input size 3, output size 5
    model.add_layer(new DenseLayer(5, 2));  // Hidden layer: input size 5, output size 2

    // Define the loss function
    MSELoss loss_fn;

    // Dummy data (for example purposes)
    Tensor input({10, 3}, { /* 30 random floats */ });
    Tensor target({10, 2}, { /* 20 random target floats */ });

    // Training loop
    int epochs = 1000;
    float learning_rate = 0.01;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward pass
        Tensor output = model.forward(input);

        // Compute the loss
        Tensor loss = loss_fn.forward(output, target);

        // Backward pass (propagate gradients)
        Tensor loss_grad = loss_fn.backward(output, target);
        model.backward(loss_grad);

        // Update weights
        model.update_weights(learning_rate);

        // Print loss every 100 epochs
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << " - Loss: " << loss.at({0}) << std::endl;
        }
    }

    return 0;
}
