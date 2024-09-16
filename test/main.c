#include <stdio.h>
#include "NNN/nnn.h"
#include "NNN/train.h"

int main(void) {
    NN_Model* model = nn_create_model();

    nn_add_layer(
        model,
        nn_create_layer(6, NN_INPUT));  // Bit 1 / Bit 2 / AND / OR / XOR / NAND
    nn_add_layer(model, nn_create_layer(8, NN_RELU));
    nn_add_layer(model, nn_create_layer(10, NN_RELU));
    nn_add_layer(model, nn_create_layer(4, NN_RELU));
    nn_add_layer(model, nn_create_layer(1, NN_SIGMOID));

    nn_init_model(model);

    NN_TrainingSet* training_set = nn_create_training_set();
    nn_add_pair(training_set, nn_create_training_pair(
                                  (double[]){0, 0, 1, 0, 0, 0}, (double[]){0}));
    nn_add_pair(training_set, nn_create_training_pair(
                                  (double[]){1, 0, 1, 0, 0, 0}, (double[]){0}));
    nn_add_pair(training_set, nn_create_training_pair(
                                  (double[]){1, 1, 1, 0, 0, 0}, (double[]){1}));
    nn_add_pair(training_set, nn_create_training_pair(
                                  (double[]){0, 1, 1, 0, 0, 0}, (double[]){0}));

    nn_add_pair(training_set, nn_create_training_pair(
                                  (double[]){0, 0, 0, 1, 0, 0}, (double[]){0}));
    nn_add_pair(training_set, nn_create_training_pair(
                                  (double[]){1, 0, 0, 1, 0, 0}, (double[]){1}));
    nn_add_pair(training_set, nn_create_training_pair(
                                  (double[]){1, 1, 0, 1, 0, 0}, (double[]){1}));
    nn_add_pair(training_set, nn_create_training_pair(
                                  (double[]){0, 1, 0, 1, 0, 0}, (double[]){1}));

    nn_add_pair(training_set, nn_create_training_pair(
                                  (double[]){0, 0, 0, 0, 1, 0}, (double[]){0}));
    nn_add_pair(training_set, nn_create_training_pair(
                                  (double[]){1, 0, 0, 0, 1, 0}, (double[]){1}));
    nn_add_pair(training_set, nn_create_training_pair(
                                  (double[]){1, 1, 0, 0, 1, 0}, (double[]){0}));
    nn_add_pair(training_set, nn_create_training_pair(
                                  (double[]){0, 1, 0, 0, 1, 0}, (double[]){1}));

    nn_add_pair(training_set, nn_create_training_pair(
                                  (double[]){0, 0, 0, 0, 0, 1}, (double[]){1}));
    nn_add_pair(training_set, nn_create_training_pair(
                                  (double[]){1, 0, 0, 0, 0, 1}, (double[]){1}));
    nn_add_pair(training_set, nn_create_training_pair(
                                  (double[]){1, 1, 0, 0, 0, 1}, (double[]){0}));
    nn_add_pair(training_set, nn_create_training_pair(
                                  (double[]){0, 1, 0, 0, 0, 1}, (double[]){1}));

    NN_TrainConfig config = {.loss = &NN_MSE,
                             .learning_rate = 0.02,
                             .epochs = 500,
                             .training_set = training_set,
                             .validation_set = NULL};

    nn_train(model, &config);

    printf("0 AND 0 = %f\n",
           nn_predict(model, (double[]){0, 0, 1, 0, 0, 0})[0]);

    printf("1 AND 0 = %f\n",
           nn_predict(model, (double[]){1, 0, 1, 0, 0, 0})[0]);

    printf("1 AND 1 = %f\n",
           nn_predict(model, (double[]){1, 1, 1, 0, 0, 0})[0]);

    printf("0 AND 1 = %f\n",
           nn_predict(model, (double[]){0, 1, 1, 0, 0, 0})[0]);

    printf("0 NAND 0 = %f\n",
           nn_predict(model, (double[]){0, 0, 0, 0, 0, 1})[0]);

    printf("1 NAND 0 = %f\n",
           nn_predict(model, (double[]){1, 0, 0, 0, 0, 1})[0]);

    printf("1 NAND 1 = %f\n",
           nn_predict(model, (double[]){1, 1, 0, 0, 0, 1})[0]);

    printf("0 NAND 1 = %f\n",
           nn_predict(model, (double[]){0, 1, 0, 0, 0, 1})[0]);

    printf("0 OR 0 = %f\n", nn_predict(model, (double[]){0, 0, 0, 1, 0, 0})[0]);

    printf("1 OR 0 = %f\n", nn_predict(model, (double[]){1, 0, 0, 1, 0, 0})[0]);

    printf("1 OR 1 = %f\n", nn_predict(model, (double[]){1, 1, 0, 1, 0, 0})[0]);

    printf("0 OR 1 = %f\n", nn_predict(model, (double[]){0, 1, 0, 1, 0, 0})[0]);

    printf("0 XOR 0 = %f\n",
           nn_predict(model, (double[]){0, 0, 0, 0, 1, 0})[0]);

    printf("1 XOR 0 = %f\n",
           nn_predict(model, (double[]){1, 0, 0, 0, 1, 0})[0]);

    printf("1 XOR 1 = %f\n",
           nn_predict(model, (double[]){1, 1, 0, 0, 1, 0})[0]);

    printf("0 XOR 1 = %f\n",
           nn_predict(model, (double[]){0, 1, 0, 0, 1, 0})[0]);

    return 0;
}
