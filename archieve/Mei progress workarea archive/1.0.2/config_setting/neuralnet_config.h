#pragma once
#include <vector>
#include <functional>
#include <string>

enum ActivationType { RELU, LEAKY_RELU, TANH, SIGMOID };
enum LossType { MSE, CROSS_ENTROPY };
enum OptimizerType { SGD, ADAM };

struct LayerConfig {
    size_t num_neurons;
    ActivationType activation;
    double dropout_keep_prob = 1.0;
};

struct NeuralConfig {
    std::vector<LayerConfig> layers;
    double learning_rate = 0.01;
    double decay_rate = 0.0;
    std::vector<double> beta = {0.9, 0.999};
    OptimizerType optimizer = ADAM;
    LossType loss = MSE;
    int step_size = 1000;
};
