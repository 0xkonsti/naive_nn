# naive_nn

A minimal feedforward neural network library written in C11 with no external dependencies. Supports configurable activation functions, MSE loss, and modular model building via a doubly-linked layer structure.

## Project structure

```
naive_nn/
├── include/NNN/    # Public API headers
│   ├── nnn.h       # Umbrella header
│   ├── activation.h
│   ├── layer.h
│   ├── loss.h
│   ├── model.h
│   └── train.h
├── src/            # Implementation
│   ├── nnn.c
│   ├── activation.c
│   ├── layer.c
│   ├── loss.c
│   ├── model.c
│   ├── train.c
│   ├── util.h
│   └── util.c
├── test/
│   └── main.c      # Example: learns AND/OR/XOR/NAND from scratch
├── CMakeLists.txt
├── Makefile
└── README.md
```

## Prerequisites

- **C11 compiler** (GCC, Clang, MSVC)
- **CMake** >= 3.10 (or use the provided Makefile)
- **make** (on Linux/macOS) or MSBuild (on Windows with Visual Studio)

## Building

### CMake

```sh
mkdir build && cd build
cmake ..
cmake --build .
```

### Make (Linux/macOS)

```sh
make debug      # Debug build
make release    # Release build
make run-debug  # Build & run
```

The resulting `main` executable trains a network on logic gate data and prints predictions.

## Quickstart

```c
#include "NNN/nnn.h"
#include "NNN/train.h"

NN_Model* model = nn_create_model();

nn_add_layer(model, nn_create_layer(2, NN_INPUT));   // 2 inputs
nn_add_layer(model, nn_create_layer(4, NN_RELU));    // hidden layer
nn_add_layer(model, nn_create_layer(1, NN_SIGMOID)); // output

nn_init_model(model);

// Training data
NN_TrainingSet* ts = nn_create_training_set();
nn_add_pair(ts, nn_create_training_pair(
    (double[]){0, 0}, (double[]){0}));
nn_add_pair(ts, nn_create_training_pair(
    (double[]){1, 0}, (double[]){1}));
// ... add more pairs

NN_TrainConfig config = {
    .loss = &NN_MSE,
    .learning_rate = 0.1,
    .epochs = 1000,
    .training_set = ts,
};

nn_train(model, &config);

double* result = nn_predict(model, (double[]){1, 0});
printf("%f\n", result[0]);

nn_destroy_training_set(ts);
nn_destroy_model(model);
```

## API overview

| Function | Purpose |
|---|---|
| `nn_create_layer(n, activation)` | Create a layer with `n` neurons |
| `nn_add_layer(model, layer)` | Append a layer to the model |
| `nn_init_model(model)` | Randomize all weights and biases in [-1, 1] |
| `nn_forward(model)` | Run a forward pass through the network |
| `nn_predict(model, input)` | `set_input` + `forward` + `get_output` |
| `nn_backward(model, target, lr, loss)` | Backpropagate gradients and update parameters |
| `nn_train(model, config)` | Train over multiple epochs with progress output |

### Activations

`NN_SIGMOID`, `NN_RELU`, `NN_TANH`, `NN_LEAKY_RELU`, `NN_INPUT` (identity)

### Loss

`NN_MSE` (Mean Squared Error)

## Notes

- Weights and biases are initialized uniformly in [-1, 1] via `rand()`. Seed with `srand()` for non-deterministic runs.
- `nn_add_pair` takes ownership of the `NN_TrainingPair*` — the caller must not free it manually.
- The training data arrays (`input`/`target`) must remain valid for the lifetime of the training set.
