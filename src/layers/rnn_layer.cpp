class RNNLayer {
public:
    RNNLayer(int input_size, int hidden_size) 
        : input_size(input_size), hidden_size(hidden_size) {
        // Initialize weights and biases
        W_h = Tensor({input_size, hidden_size});   // Input to hidden
        U_h = Tensor({hidden_size, hidden_size});  // Hidden to hidden
        b_h = Tensor({hidden_size});               // Bias
    }

    Tensor forward(const std::vector<Tensor>& inputs) {
        // Initialize hidden state
        Tensor h = Tensor({hidden_size});
        h.fill(0.0);

        // Iterate over the input sequence
        for (const Tensor& x : inputs) {
            h = tanh(W_h.matmul(x) + U_h.matmul(h) + b_h);
        }

        return h; // Final hidden state as the output
    }

private:
    int input_size, hidden_size;
    Tensor W_h, U_h, b_h;
};

class LSTMLayer {
public:
    LSTMLayer(int input_size, int hidden_size) 
        : input_size(input_size), hidden_size(hidden_size) {
        // Initialize weights for gates
        W_f = Tensor({input_size, hidden_size});
        U_f = Tensor({hidden_size, hidden_size});
        b_f = Tensor({hidden_size});

        W_i = Tensor({input_size, hidden_size});
        U_i = Tensor({hidden_size, hidden_size});
        b_i = Tensor({hidden_size});

        W_o = Tensor({input_size, hidden_size});
        U_o = Tensor({hidden_size, hidden_size});
        b_o = Tensor({hidden_size});

        W_c = Tensor({input_size, hidden_size});
        U_c = Tensor({hidden_size, hidden_size});
        b_c = Tensor({hidden_size});
    }

    Tensor forward(const std::vector<Tensor>& inputs) {
        // Initialize hidden and cell state
        Tensor h = Tensor({hidden_size});
        Tensor C = Tensor({hidden_size});
        h.fill(0.0);
        C.fill(0.0);

        for (const Tensor& x : inputs) {
            // Forget gate
            Tensor f_t = sigmoid(W_f.matmul(x) + U_f.matmul(h) + b_f);
            // Input gate
            Tensor i_t = sigmoid(W_i.matmul(x) + U_i.matmul(h) + b_i);
            // Output gate
            Tensor o_t = sigmoid(W_o.matmul(x) + U_o.matmul(h) + b_o);
            // Candidate cell state
            Tensor C_tilde = tanh(W_c.matmul(x) + U_c.matmul(h) + b_c);
            // Update cell state
            C = f_t * C + i_t * C_tilde;
            // Update hidden state
            h = o_t * tanh(C);
        }

        return h;
    }

private:
    int input_size, hidden_size;
    Tensor W_f, U_f, b_f;
    Tensor W_i, U_i, b_i;
    Tensor W_o, U_o, b_o;
    Tensor W_c, U_c, b_c;
};

class GRULayer {
public:
    GRULayer(int input_size, int hidden_size) 
        : input_size(input_size), hidden_size(hidden_size) {
        // Initialize weights
        W_z = Tensor({input_size, hidden_size});
        U_z = Tensor({hidden_size, hidden_size});
        b_z = Tensor({hidden_size});

        W_r = Tensor({input_size, hidden_size});
        U_r = Tensor({hidden_size, hidden_size});
        b_r = Tensor({hidden_size});

        W_h = Tensor({input_size, hidden_size});
        U_h = Tensor({hidden_size, hidden_size});
        b_h = Tensor({hidden_size});
    }

    Tensor forward(const std::vector<Tensor>& inputs) {
        // Initialize hidden state
        Tensor h = Tensor({hidden_size});
        h.fill(0.0);

        for (const Tensor& x : inputs) {
            // Update gate
            Tensor z_t = sigmoid(W_z.matmul(x) + U_z.matmul(h) + b_z);
            // Reset gate
            Tensor r_t = sigmoid(W_r.matmul(x) + U_r.matmul(h) + b_r);
            // Candidate hidden state
            Tensor h_tilde = tanh(W_h.matmul(x) + U_h.matmul(r_t * h) + b_h);
            // Update hidden state
            h = (1 - z_t) * h + z_t * h_tilde;
        }

        return h;
    }

private:
    int input_size, hidden_size;
    Tensor W_z, U_z, b_z;
    Tensor W_r, U_r, b_r;
    Tensor W_h, U_h, b_h;
};
