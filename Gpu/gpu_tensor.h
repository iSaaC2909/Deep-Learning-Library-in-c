#ifndef GPU_TENSOR_H
#define GPU_TENSOR_H

#include <vector>
#include <cuda_runtime.h>

class GPUTensor {
public:
    GPUTensor(const std::vector<int>& dims);
    ~GPUTensor();

    void allocate();
    void copyFromHost(const std::vector<float>& data);
    void copyToHost(std::vector<float>& data) const;

    // CUDA kernel functions
    static void addKernel(const GPUTensor& a, const GPUTensor& b, GPUTensor& result);
    static void matmulKernel(const GPUTensor& a, const GPUTensor& b, GPUTensor& result);

private:
    float* data_;         // Pointer to GPU memory
    std::vector<int> dims_; // Dimensions of the tensor
    size_t size_;         // Total size in bytes
};

#endif // GPU_TENSOR_H
