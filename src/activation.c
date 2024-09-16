#include <math.h>
#include "NNN/nnn.h"

double __sigmoid(double const x) {
    return 1.0 / (1.0 + exp(-x));
}

double __sigmoid_derivative(double const x) {
    return x * (1.0 - x);
}

NN_Activation const NN_SIGMOID = {
    .function = __sigmoid,
    .derivative = __sigmoid_derivative,
};

double __relu(double const x) {
    return x > 0 ? x : 0;
}

double __relu_derivative(double const x) {
    return x > 0 ? 1 : 0;
}

NN_Activation const NN_RELU = {
    .function = __relu,
    .derivative = __relu_derivative,
};

double __tanh(double const x) {
    return tanh(x);
}

double __tanh_derivative(double const x) {
    return 1 - x * x;
}

NN_Activation const NN_TANH = {
    .function = __tanh,
    .derivative = __tanh_derivative,
};

double __leaky_relu(double const x) {
    return x > 0 ? x : NN_LEAKY_RELU_NEGATIVE_SLOPE * x;
}

double __leaky_relu_derivative(double const x) {
    return x > 0 ? 1 : NN_LEAKY_RELU_NEGATIVE_SLOPE;
}

NN_Activation const NN_LEAKY_RELU = {
    .function = __leaky_relu,
    .derivative = __leaky_relu_derivative,
};

double __input(double const x) {
    return x;
}

double __input_derivative(double const x) {
    return x;
}

NN_Activation const NN_INPUT = {
    .function = __input,
    .derivative = __input_derivative,
};
