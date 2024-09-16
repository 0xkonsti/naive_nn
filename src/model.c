#include "NNN/model.h"
#include <stdlib.h>

NN_Model* nn_create_model(void) {
    NN_Model* model = malloc(sizeof(NN_Model));
    model->input = NULL;
    model->output = NULL;
    return model;
}

void nn_destroy_model(NN_Model* model) {
    for (NN_Layer* layer = model->input; layer != NULL; layer = layer->next) {
        nn_destroy_layer(layer);
    }

    free(model);
}

void nn_add_layer(NN_Model* model, NN_Layer* layer) {
    if (model->input == NULL) {
        model->input = layer;
    }

    if (model->output != NULL) {
        model->output->connect(model->output, layer);
    }

    model->output = layer;
}

void nn_init_model(NN_Model const* model) {
    for (NN_Layer* layer = model->input; layer != NULL; layer = layer->next) {
        nn_init_layer(layer);
    }
}

void nn_forward(NN_Model const* model) {
    for (NN_Layer const* layer = model->input; layer != NULL;
         layer = layer->next) {
        layer->forward(layer);
    }
}

void nn_backward(NN_Model const* model, double const* target,
                 double const learning_rate, NN_Loss const* loss) {
    for (uint32_t i = 0; i < model->output->neuron_count; i++) {
        model->output->weight_gradients[i] =
            loss->derivative(model->output->neurons[i], target[i]);
    }

    for (NN_Layer const* layer = model->output; layer != NULL;
         layer = layer->prev) {
        layer->backward(layer, learning_rate);
    }
}

void nn_set_input(NN_Model const* model, double const* input) {
    NN_Layer const* input_layer = model->input;
    for (uint32_t i = 0; i < input_layer->neuron_count; i++) {
        input_layer->neurons[i] = input[i];
    }
}

double* nn_get_output(NN_Model const* model) {
    return model->output->neurons;
}

double* nn_predict(NN_Model const* model, double const* input) {
    nn_set_input(model, input);
    nn_forward(model);
    return nn_get_output(model);
}