#include "NNN/loss.h"

double __mse_function(double const predicted, double const actual) {
    return (predicted - actual) * (predicted - actual);
}

double __mse_derivative(double const predicted, double const actual) {
    return 2 * (predicted - actual);
}

double __mse_total(double const* predicted, double const* actual,
                   size_t const size) {
    double sum = 0;
    for (size_t i = 0; i < size; i++) {
        sum += __mse_function(predicted[i], actual[i]);
    }
    return sum / size;
}

NN_Loss const NN_MSE = {
    .total = __mse_total,
    .function = __mse_function,
    .derivative = __mse_derivative,
};
