#include "gpu_tensor.h"
#include <iostream>

GPUTensor::GPUTensor(const std::vector<int>& dims) : dims_(dims) {
    size_ = 1;
    for (int dim : dims) {
        size_ *= dim;
    }
    allocate();
}

GPUTensor::~GPUTensor() {
    cudaFree(data_);
}

void GPUTensor::allocate() {
    cudaMalloc(&data_, size_ * sizeof(float));
}

void GPUTensor::copyFromHost(const std::vector<float>& data) {
    cudaMemcpy(data_, data.data(), size_ * sizeof(float), cudaMemcpyHostToDevice);
}

void GPUTensor::copyToHost(std::vector<float>& data) const {
    cudaMemcpy(data.data(), data_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
}

// Example kernel for element-wise addition
__global__ void addKernel(const float* a, const float* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

// Implementing the addition function
void GPUTensor::addKernel(const GPUTensor& a, const GPUTensor& b, GPUTensor& result) {
    int threadsPerBlock = 256;
    int blocks = (result.size_ + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocks, threadsPerBlock>>>(a.data_, b.data_, result.data_, result.size_);
    cudaDeviceSynchronize();  // Wait for GPU to finish
}
