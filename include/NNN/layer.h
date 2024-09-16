#ifndef LAYER_H
#define LAYER_H

#include <stdint.h>
#include "NNN/activation.h"

/**
 * NN_Weights
 *
 * Connection weights matrix between two layers of neurons.
 */
typedef struct {
    uint32_t left_size;
    uint32_t right_size;

    double** weights;
} NN_Weights;

NN_Weights* nn_create_weights(uint32_t left_size, uint32_t right_size);
void nn_destroy_weights(NN_Weights* weights);

/**
 * NN_Layer
 *
 * A layer of neurons in a neural network.
 * The layer can be connected to another layer of neurons with weights.
 */
typedef struct NN_Layer NN_Layer;
struct NN_Layer {
    uint32_t neuron_count;
    double* neurons;
    double* biases;

    NN_Activation activation;

    NN_Layer* next;
    NN_Layer* prev;

    NN_Weights* weights;
    double* weight_gradients;

    void (*forward)(NN_Layer const*);
    void (*backward)(NN_Layer const*, double);
    void (*connect)(NN_Layer*, NN_Layer*);
};

/**
 *  create_nn_layer
 *
 *  Create a new layer of neurons.
 *
 *  @param neuron_count  The number of neurons in the layer.
 *  @param activation    The activation function of the neurons.
 *  @return              A pointer to the new layer.
 */
NN_Layer* nn_create_layer(uint32_t neuron_count, NN_Activation activation);

/**
 *  destroy_nn_layer
 *
 *  Destroy a layer of neurons.
 *
 *  @param layer  The layer to destroy.
 */
void nn_destroy_layer(NN_Layer* layer);

/**
 *  init_nn_layer
 *
 *  Initialize a layer of neurons with random weights and biases.
 *
 *  @param layer  The layer to initialize.
 */
void nn_init_layer(NN_Layer const* layer);

#endif  // LAYER_H
