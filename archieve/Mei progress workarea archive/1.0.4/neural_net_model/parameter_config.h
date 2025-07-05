/* Project: NEURA 
NEURA is an artificial neuron and network development research project 
focused on artificial intelligence for data analysis. The goal is to find 
the optimal neural network structure and improve artificial neurons for 
efficiency and speed.

Future projects:
- Project Mei: Neuron investigation project
- Project Raiden: Network development 
- Hardware integration: After successfully developing the artificial 
  neuron network, the project will transition into hardware implementation.

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

#ifndef PARAMETER_CONFIG_H
#define PARAMETER_CONFIG_H

#pragma once

#include <iostream>
#include <vector>
#include "../data_tools/cdata_tools.h"

using namespace std;

struct DataConfig {
    vector<double> inputs = {5, 2, 4, 4, 1, 3, 8, 6, 7};
    vector<double> targets = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    vector<double> label = {0, 0, 0, 0, 1, 0, 0, 0, 0};
    vector<double> keep_prob = {0.2, 0.3, 0.4, 0.5, 0.95};
    vector<double> normalized_inputs = Data(targets).targetdataset_normalization(SYMMETRIC, inputs);
    vector<double> normalized_targets = Data(targets).dataset_normalization(SYMMETRIC);
};

struct Hyperparameters {
    vector<size_t> num_neurons = {9, 9, 9, 9, 9};
    double learning_rate = 1e-2;
    double decay_rate = 1e-6;
    vector<double> beta = {0.9, 0.9999};
    int step_size = 1000;
    double threshold_error = 0.05;
};

#endif // PARAMETER_CONFIG_H
