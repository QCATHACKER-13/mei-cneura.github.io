/* Project: NEURA 
NEURA is an artificial neuron and network development research project 
focused on artificial intelligence for data analysis. The goal is to find 
the optimal neural network structure and improve artificial neurons for 
efficiency and speed.

Future projects:
- Project Mei: Neuron investigation project

Developer: Christopher Emmanuelle J. Visperas, Applied Physics Researcher*/

/*
This is data parameter configuration for the neural network.
This file contains the necessary configurations for the neural network parameters,
including the number of neurons, input data, learning rate, decay rate, beta values,
activation functions, learning rate schedules, optimizers, and loss functions.
It is used to initialize the neural network and set up the training process.
This file is part of the NEURA project, which focuses on developing artificial neurons
and networks for data analysis and artificial intelligence applications.
It is designed to be used with the C++17 language standard version.
*/

#ifndef TRANSFORMER_PARAMETER_CONFIG_H
#define TRANSFORMER_PARAMETER_CONFIG_H

#pragma once

#include <iostream>
#include <vector>
#include "../data_tools/cdata_tools.h"

using namespace std;

struct DataConfig {
    vector<double> inputs = {5, 2, 4, 4, 1, 3, 8, 6, 7};
    vector<double> decoder_target = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    vector<double> label = {0, 0, 0, 0, 1, 0, 0, 0, 0};
    vector<double> encoder_target = {2, 5, 7};
    double labeled_data = 5;
    vector<double> keep_prob = {0.35, 0.65, 0.95};
    vector<double> normalized_inputs = Data(decoder_target).targetdataset_normalization(SYMMETRIC, inputs);
    vector<double> normalized_targets = Data(decoder_target).dataset_normalization(SYMMETRIC);
};

struct Hyperparameters {
    vector<size_t> num_neuron_decoder = {3, 6, 9};
    vector<size_t> num_neuron_encoder = {9, 6, 3};
    double learning_rate = 1e-2;
    double decay_rate = 1e-6;
    vector<double> beta = {0.9, 0.9999};
    int step_size = 1000;
    double threshold_error = 0.05;
};

#endif // PARAMETER_CONFIG_H
