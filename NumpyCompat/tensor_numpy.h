#ifndef TENSOR_NUMPY_H
#define TENSOR_NUMPY_H

#include <vector>
#include <Python.h>
#include "gpu_tensor.h" // Or your main Tensor class

class TensorNumpy {
public:
    static PyObject* to_numpy(const GPUTensor& tensor);
    static void from_numpy(GPUTensor& tensor, PyObject* np_array);
};

#endif // TENSOR_NUMPY_H
