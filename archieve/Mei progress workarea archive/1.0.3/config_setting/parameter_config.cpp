#include <iostream>
#include "parameter_config.h"

int main() {
    DataConfig data;
    Hyperparameters hyper;

    std::cout << "DataConfig.inputs: ";
    for (auto v : data.inputs) std::cout << v << " ";
    std::cout << "\nDataConfig.targets: ";
    for (auto v : data.targets) std::cout << v << " ";
    std::cout << "\nDataConfig.label: ";
    for (auto v : data.label) std::cout << v << " ";
    std::cout << "\nDataConfig.keep_prob: ";
    for (auto v : data.keep_prob) std::cout << v << " ";
    std::cout << "\nDataConfig.normalized_inputs: ";
    for (auto v : data.normalized_inputs) std::cout << v << " ";
    std::cout << "\nDataConfig.normalized_targets: ";
    for (auto v : data.normalized_targets) std::cout << v << " ";
    std::cout << "\n";

    std::cout << "Hyperparameters.num_neurons: ";
    for (auto v : hyper.num_neurons) std::cout << v << " ";
    std::cout << "\nHyperparameters.learning_rate: " << hyper.learning_rate;
    std::cout << "\nHyperparameters.decay_rate: " << hyper.decay_rate;
    std::cout << "\nHyperparameters.beta: ";
    for (auto v : hyper.beta) std::cout << v << " ";
    std::cout << "\nHyperparameters.step_size: " << hyper.step_size << std::endl;

    return 0;
}