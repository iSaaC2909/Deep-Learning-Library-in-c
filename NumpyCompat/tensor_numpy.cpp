#include "tensor_numpy.h"
#include <numpy/arrayobject.h>

PyObject* TensorNumpy::to_numpy(const GPUTensor& tensor) {
    npy_intp dims[Tensor::max_dimensions]; // Assuming you have a max_dimensions constant
    for (size_t i = 0; i < tensor.dims_.size(); ++i) {
        dims[i] = tensor.dims_[i];
    }

    // Create a NumPy array
    PyObject* np_array = PyArray_SimpleNew(1, dims, NPY_FLOAT);
    
    // Copy data from GPU to host
    std::vector<float> host_data(tensor.size_);
    tensor.copyToHost(host_data);

    // Copy data into NumPy array
    std::memcpy(PyArray_DATA((PyArrayObject*)np_array), host_data.data(), tensor.size_ * sizeof(float));
    return np_array;
}

void TensorNumpy::from_numpy(GPUTensor& tensor, PyObject* np_array) {
    // Assume np_array is a float type array
    float* data = static_cast<float*>(PyArray_DATA((PyArrayObject*)np_array));

    // Copy data from NumPy to GPU tensor
    std::vector<float> host_data(data, data + tensor.size_);
    tensor.copyFromHost(host_data);
}
