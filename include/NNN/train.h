#ifndef NN_TRAIN_H
#define NN_TRAIN_H

#include "NNN/loss.h"
#include "NNN/model.h"

typedef struct {
    double* input;
    double* target;
} NN_TrainingPair;

NN_TrainingPair* nn_create_training_pair(double* input, double* target);
void nn_destroy_training_pair(NN_TrainingPair* pair);

typedef struct {
    NN_TrainingPair* pairs;
    int num_pairs;
} NN_TrainingSet;

NN_TrainingSet* nn_create_training_set(void);
void nn_destroy_training_set(NN_TrainingSet* set);

void nn_add_pair(NN_TrainingSet* set, NN_TrainingPair const* pair);

typedef struct {
    NN_Loss const* loss;
    double learning_rate;
    int epochs;

    NN_TrainingSet* training_set;
    NN_TrainingSet* validation_set;
} NN_TrainConfig;

void nn_train(NN_Model const* model, NN_TrainConfig const* config);

#endif  // NN_TRAIN_H
