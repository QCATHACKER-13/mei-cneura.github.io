#include "../decoder_model/decoder_neural_config.h"
#include "../decoder_model/decoder_parameter_config.h"
#include "../building_model/decoder_build.h"
#include <iostream>

int main() {
    try {
        DataConfig data_config;           // Uses default constructor/data
        Hyperparameters hyper_config;     // Uses default constructor
        NeuralConfig neural_config;       // Uses default constructor

        // Build the neural network
    Neural net = DecoderModel::build(data_config, hyper_config, neural_config);

        net.enforce_learning(hyper_config.step_size, hyper_config.threshold_error);

        std::cout << "Neural network built successfully!" << std::endl;
        // Optionally, call methods on 'net' to verify it's working
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}