#include <iostream>
#include "neural_config.h"

int main() {
    NeuralConfig config;

    std::cout << "NeuralConfig default values:" << std::endl;
    std::cout << "Activation Functions: ";
    for (auto af : config.actFunc) std::cout << af << " ";
    std::cout << "\nLearning Rate Schedule: " << config.lr_schedule;
    std::cout << "\nDistribution: " << config.distribution;
    std::cout << "\nOptimizer: " << config.opt;
    std::cout << "\nLoss Function: " << config.lossFunc << std::endl;

    return 0;
}