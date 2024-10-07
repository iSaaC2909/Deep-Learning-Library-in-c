#ifndef TENSOR_EXCEPTIONS_H
#define TENSOR_EXCEPTIONS_H

#include <exception>
#include <string>

// Base exception class for Tensor-related errors
class TensorException : public std::exception {
public:
    explicit TensorException(const std::string& message) : msg_(message) {}
    virtual const char* what() const noexcept override {
        return msg_.c_str();
    }
private:
    std::string msg_;
};

// Exception for dimension mismatches
class DimensionMismatchException : public TensorException {
public:
    DimensionMismatchException(const std::string& message)
        : TensorException("DimensionMismatchException: " + message) {}
};

// Exception for out-of-bounds indexing
class IndexOutOfBoundsException : public TensorException {
public:
    IndexOutOfBoundsException(const std::string& message)
        : TensorException("IndexOutOfBoundsException: " + message) {}
};

// Exception for division by zero
class DivisionByZeroException : public TensorException {
public:
    DivisionByZeroException(const std::string& message)
        : TensorException("DivisionByZeroException: " + message) {}
};

// Exception for invalid operations
class InvalidOperationException : public TensorException {
public:
    InvalidOperationException(const std::string& message)
        : TensorException("InvalidOperationException: " + message) {}
};

// Exception for division by zero
class DivisionByZeroException : public TensorException {
public:
    DivisionByZeroException(const std::string& message)
        : TensorException("DivisionByZeroException: " + message) {}
};

// Exception for invalid tensor operations
class InvalidTensorOperationException : public TensorException {
public:
    InvalidTensorOperationException(const std::string& message)
        : TensorException("InvalidTensorOperationException: " + message) {}
};


#endif // TENSOR_EXCEPTIONS_H
