#include "NNN/train.h"
#include <stdio.h>
#include <stdlib.h>
#include "NNN/model.h"

NN_TrainingPair* nn_create_training_pair(double* input, double* target) {
    NN_TrainingPair* pair = malloc(sizeof(NN_TrainingPair));
    pair->input = input;
    pair->target = target;
    return pair;
}

void nn_destroy_training_pair(NN_TrainingPair* pair) {
    free(pair);
}

NN_TrainingSet* nn_create_training_set(void) {
    NN_TrainingSet* set = malloc(sizeof(NN_TrainingSet));
    set->pairs = NULL;
    set->num_pairs = 0;
    return set;
}

void nn_destroy_training_set(NN_TrainingSet* set) {
    for (int i = 0; i < set->num_pairs; i++) {
        nn_destroy_training_pair(&set->pairs[i]);
    }

    free(set->pairs);
    free(set);
}

void nn_add_pair(NN_TrainingSet* set, NN_TrainingPair const* pair) {
    set->num_pairs++;
    set->pairs = realloc(set->pairs, set->num_pairs * sizeof(NN_TrainingPair));
    set->pairs[set->num_pairs - 1] = *pair;
}

void __single_pair(NN_Model const* model, NN_TrainingPair const* pair,
                   double const learning_rate, NN_Loss const* loss) {
    nn_set_input(model, pair->input);
    nn_forward(model);
    nn_backward(model, pair->target, learning_rate, loss);
}

void nn_train(NN_Model const* model, NN_TrainConfig const* config) {
    printf("Training model...\n");

    for (int epoch = 0; epoch < config->epochs; epoch++) {
        printf("Epoch %010d / %010d :", epoch + 1, config->epochs);
        double avg_loss = 0.0;
        for (int i = 0; i < config->training_set->num_pairs; i++) {
            __single_pair(model, &config->training_set->pairs[i],
                          config->learning_rate, config->loss);
            avg_loss += config->loss->total(
                model->output->neurons, config->training_set->pairs[i].target,
                model->output->neuron_count);
        }
        avg_loss /= config->training_set->num_pairs;
        printf(" Avg. Loss: %f\r", avg_loss);
    }

    printf("Training complete.\n");
}
