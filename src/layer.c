#include "NNN/layer.h"
#include <stdio.h>
#include <stdlib.h>
#include "util.h"

NN_Weights* nn_create_weights(uint32_t const left_size,
                              uint32_t const right_size) {
    NN_Weights* weights = malloc(sizeof(NN_Weights));
    if (weights == NULL) {
        return NULL;
    }

    weights->left_size = left_size;
    weights->right_size = right_size;

    weights->weights = (double**)malloc(left_size * sizeof(double*));
    if (weights->weights == NULL) {
        free(weights);
        return NULL;
    }

    for (uint32_t i = 0; i < left_size; i++) {
        weights->weights[i] = (double*)malloc(right_size * sizeof(double));
        if (weights->weights[i] == NULL) {
            for (uint32_t j = 0; j < i; j++) {
                free(weights->weights[j]);
            }
            free(weights->weights);
            free(weights);
            return NULL;
        }
    }

    return weights;
}

void nn_destroy_weights(NN_Weights* weights) {
    for (uint32_t i = 0; i < weights->left_size; i++) {
        free(weights->weights[i]);
    }
    free(weights->weights);
    free(weights);
}

void __forward(NN_Layer const* layer) {
    if (layer->next == NULL || layer->weights == NULL) {
        return;
    }

    for (uint32_t i = 0; i < layer->next->neuron_count; i++) {
        double sum = 0.0;
        for (uint32_t j = 0; j < layer->neuron_count; j++) {
            sum += layer->neurons[j] * layer->weights->weights[j][i];
        }
        layer->next->neurons[i] =
            layer->next->activation.function(sum + layer->next->biases[i]);
    }
}

void __backward(NN_Layer const* layer, double const learning_rate) {
    if (layer->prev == NULL || layer->weights == NULL) {
        return;
    }

    for (uint32_t i = 0; i < layer->neuron_count; i++) {
        double sum = 0.0;
        for (uint32_t j = 0; j < layer->next->neuron_count; j++) {
            sum += layer->next->weight_gradients[j] *
                   layer->weights->weights[i][j];
        }
        layer->weight_gradients[i] =
            sum * layer->activation.derivative(layer->neurons[i]);
    }

    for (uint32_t i = 0; i < layer->neuron_count; i++) {
        for (uint32_t j = 0; j < layer->next->neuron_count; j++) {
            layer->weights->weights[i][j] -= layer->next->weight_gradients[j] *
                                             layer->neurons[i] * learning_rate;
        }
    }

    for (uint32_t i = 0; i < layer->next->neuron_count; i++) {
        layer->next->biases[i] -=
            layer->next->weight_gradients[i] * learning_rate;
    }
}

void __connect(NN_Layer* left, NN_Layer* right) {
    if (left->next || right->prev) {
        return;
    }

    left->next = right;
    right->prev = left;

    left->weights = nn_create_weights(left->neuron_count, right->neuron_count);
}

NN_Layer* nn_create_layer(uint32_t const neuron_count,
                          NN_Activation const activation) {
    NN_Layer* layer = malloc(sizeof(NN_Layer));
    if (layer == NULL) {
        return NULL;
    }

    layer->neuron_count = neuron_count;
    layer->neurons = (double*)malloc(neuron_count * sizeof(double));
    if (layer->neurons == NULL) {
        free(layer);
        return NULL;
    }

    layer->biases = (double*)malloc(neuron_count * sizeof(double));
    if (layer->biases == NULL) {
        free(layer->neurons);
        free(layer);
        return NULL;
    }

    layer->activation = activation;
    layer->next = NULL;
    layer->prev = NULL;
    layer->weights = NULL;
    layer->weight_gradients = (double*)malloc(neuron_count * sizeof(double));

    layer->forward = __forward;
    layer->backward = __backward;
    layer->connect = __connect;

    return layer;
}

void nn_destroy_layer(NN_Layer* layer) {
    if (layer->weights) {
        nn_destroy_weights(layer->weights);
    }
    free(layer->biases);
    free(layer->neurons);
    free(layer);
}

void nn_init_layer(NN_Layer const* layer) {
    for (uint32_t i = 0; i < layer->neuron_count; i++) {
        layer->neurons[i] = 0.0;
        layer->biases[i] = randrange(-1.0, 1.0);
        layer->weight_gradients[i] = 0.0;
    }

    if (layer->weights) {
        for (uint32_t i = 0; i < layer->weights->left_size; i++) {
            for (uint32_t j = 0; j < layer->weights->right_size; j++) {
                layer->weights->weights[i][j] = randrange(-1.0, 1.0);
            }
        }
    }
}
