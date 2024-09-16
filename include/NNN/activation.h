#ifndef ACTIVATION_H
#define ACTIVATION_H

/**
 *  NN_Activation
 *
 * Activation function and its derivative.
 */
typedef struct {
    double (*function)(double);
    double (*derivative)(double);
} NN_Activation;

/**
 *  NN_SIGMOID
 *
 * Sigmoid is a non-linear activation function. It squashes the input to the
 * range of [0, 1].
 */
extern NN_Activation const NN_SIGMOID;

/**
 *  NN_RELU
 *
 * Rectified Linear Unit (ReLU) is a non-linear activation function. It
 * squashes the input to the range of [0, +inf).
 */
extern NN_Activation const NN_RELU;

/**
 *  NN_TANH
 *
 * Hyperbolic Tangent (tanh) is a non-linear activation function. It squashes
 * the input to the range of [-1, 1].
 */
extern NN_Activation const NN_TANH;

#ifndef NN_LEAKY_RELU_NEGATIVE_SLOPE
#define NN_LEAKY_RELU_NEGATIVE_SLOPE 0.01
#elif NN_LEAKY_RELU_NEGATIVE_SLOPE <= 0 || NN_LEAKY_RELU_NEGATIVE_SLOPE >= 1
#error "NN_LEAKY_RELU_NEGATIVE_SLOPE must be in the range of (0, 1)"
#endif

/**
 *  NN_LEAKY_RELU
 *
 * Leaky ReLU is a non-linear activation function. It squashes the input to the
 * range of (-inf, +inf) with a small negative slope for negative values.
 *
 * To change the negative slope, define NN_LEAKY_RELU_NEGATIVE_SLOPE before
 * including this header. The default value is 0.01. The value must be in the
 * range of (0, 1).
 */
extern NN_Activation const NN_LEAKY_RELU;

/**
 * NN_INPUT
 *
 * Special activation function for the input layer. It does not apply any
 * function.
 */
extern NN_Activation const NN_INPUT;

#endif  // ACTIVATION_H
