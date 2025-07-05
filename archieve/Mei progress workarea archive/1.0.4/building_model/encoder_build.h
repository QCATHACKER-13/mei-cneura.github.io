/* Project: NEURA 
NEURA is an artificial neuron and network development research project 
focused on artificial intelligence for data analysis. The goal is to find 
the optimal neural network structure and improve artificial neurons for 
efficiency and speed.

Project Development and Innovations:
- Project Neura: Neuron investigation project
- Project Mei: Neural Net Architecture investigation project

Developer: Christopher Emmanuelle J. Visperas, Applied Physics Researcher*/


#ifndef ENCODER_BUILD_H
#define ENCODER_BUILD_H

#pragma once

#include <iostream>
#include <vector>
#include "../encoder_model/encoder_neural_config.h"
#include "../encoder_model/encoder_parameter_config.h"
#include "../core_model/cneura.h"
using namespace std;

class EncoderModel {
public:
    // Build a Neural Network from the given configuration and input data
    [[nodiscard]]
    static Neural build(
        DataConfig data_config, 
        Hyperparameters hyperpara_config, 
        NeuralConfig neural_config) 
    {
        //-----------------------Don't Edit This Part-----------------------
        if (data_config.inputs.empty()) {
            throw std::invalid_argument("DataConfig.inputs must not be empty.");
        }
        
        if (hyperpara_config.num_neurons.empty()) {
            throw std::invalid_argument("Hyperparameters.num_neurons must not be empty.");
        }
        
        // Optionally, check that the config vectors are the correct size
        // (e.g., actFunc.size() == num_neurons.size(), etc.)

        Neural neural(
            hyperpara_config.num_neurons,
            data_config.inputs,
            hyperpara_config.learning_rate,
            hyperpara_config.decay_rate,
            hyperpara_config.beta,
            neural_config.initializer,
            neural_config.distribution,
            neural_config.actFunc,
            neural_config.lr_schedule,
            neural_config.opt,
            neural_config.lossFunc
        );

        neural.set_step_size(hyperpara_config.step_size);
        neural.set_dropout(data_config.keep_prob); // Set dropout rate for regularization
        neural.set_input(data_config.normalized_inputs);
        neural.set_target(data_config.normalized_targets);
        // neural.set_softlabeling(data_config.label, 1e-6);
        neural.set_hardlabeling(data_config.label);
        //-----------------------End-----------------------

        return neural;
    }
};

#endif // NEURAL_BUILD_H
// End of neural_build.h