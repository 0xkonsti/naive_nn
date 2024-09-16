#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>
#include "NNN/layer.h"
#include "NNN/loss.h"

typedef struct {
    NN_Layer* input;
    NN_Layer* output;
} NN_Model;

NN_Model* nn_create_model(void);
void nn_destroy_model(NN_Model* model);

void nn_add_layer(NN_Model* model, NN_Layer* layer);
void nn_init_model(NN_Model const* model);

void nn_forward(NN_Model const* model);
void nn_backward(NN_Model const* model, double const* target,
                 double learning_rate, NN_Loss const* loss);

void nn_set_input(NN_Model const* model, double const* input);
double* nn_get_output(NN_Model const* model);
double* nn_predict(NN_Model const* model, double const* input);

#endif  // MODEL_H
