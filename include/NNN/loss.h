#ifndef LOSS_H
#define LOSS_H

#include <stddef.h>

/**
 *  NN_Loss
 *
 * A loss function is a function that calculates the difference between the
 * predicted and the actual values. It is used to train the neural network.
 */
typedef struct {
    double (*total)(double const*, double const*, size_t);

    double (*function)(double, double);
    double (*derivative)(double, double);
} NN_Loss;

/**
 *  NN_MSE
 *
 * Mean Squared Error (MSE) is a loss function. It calculates the average of
 * the squared differences between the predicted and the actual values.
 *
 * Typically used for regression problems.
 */
extern NN_Loss const NN_MSE;

#endif  // LOSS_H
