#ifndef RNN_LAYER_H
#define RNN_LAYER_H

#include "tensor.h"

class RNNLayer {
public:
    RNNLayer(int input_size, int hidden_size);
    Tensor forward(const std::vector<Tensor>& inputs);
    
private:
    int input_size, hidden_size;
    Tensor W_h, U_h, b_h;
};

class LSTMLayer {
public:
    LSTMLayer(int input_size, int hidden_size);
    Tensor forward(const std::vector<Tensor>& inputs);

private:
    int input_size, hidden_size;
    Tensor W_f, U_f, b_f;
    Tensor W_i, U_i, b_i;
    Tensor W_o, U_o, b_o;
    Tensor W_c, U_c, b_c;
};

class GRULayer {
public:
    GRULayer(int input_size, int hidden_size);
    Tensor forward(const std::vector<Tensor>& inputs);

private:
    int input_size, hidden_size;
    Tensor W_z, U_z, b_z;
    Tensor W_r, U_r, b_r;
    Tensor W_h, U_h, b_h;
};

#endif // RNN_LAYER_H
