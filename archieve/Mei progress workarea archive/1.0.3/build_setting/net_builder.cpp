#include "../config_setting/neural_config.h"
#include "../config_setting/parameter_config.h"
#include "../build_setting/neural_build.h"
#include <iostream>

int main() {
    try {
        DataConfig data_config;           // Uses default constructor/data
        Hyperparameters hyper_config;     // Uses default constructor
        NeuralConfig neural_config;       // Uses default constructor

        // Build the neural network
        Neural net = NetBuilder::build(data_config, hyper_config, neural_config);

        std::cout << "Neural network built successfully!" << std::endl;
        // Optionally, call methods on 'net' to verify it's working
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}