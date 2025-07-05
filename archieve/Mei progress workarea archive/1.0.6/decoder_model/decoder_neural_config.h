/* Project: NEURA 
NEURA is an artificial neuron and network development research project 
focused on artificial intelligence for data analysis. The goal is to find 
the optimal neural network structure and improve artificial neurons for 
efficiency and speed.

Future projects:
- Project Mei: Neuron investigation project
Developer: Christopher Emmanuelle J. Visperas, Applied Physics Researcher*/

/*
This is neural configuration header file for the CNEURA project.
This file contains the necessary configurations for the neural network parameters,
including activation functions, learning rate schedules, optimizers, and loss functions.
It is used to initialize the neural network and set up the training process.
This file is part of the NEURA project, which focuses on developing artificial neurons
and networks for data analysis and artificial intelligence applications.
It is designed to be used with the C++17 language standard version.
*/

#ifndef DECODER_NEURAL_CONFIG_H
#define DECODER_NEURAL_CONFIG_H

#pragma once

#include <iostream>
#include <vector>
#include "../model_options/model_option.h"

using namespace std;




struct NeuralConfig{
    vector<ACTFUNC> actFunc{
        LEAKY_RELU, LEAKY_RELU, LEAKY_RELU, LEAKY_RELU, LEAKY_RELU, LEAKY_RELU, //Hidden
        LEAKY_RELU // Output
    };

    LEARNRATE lr_schedule = LEARNRATE::ITDECAY;
    DISTRIBUTION distribution = DISTRIBUTION::UNIFORM; // Distribution type for weight initialization
    INITIALIZATION initializer = INITIALIZATION::HE;
    OPTIMIZER opt = OPTIMIZER::ADAM; // Optimizer type
    vector<LOSSFUNC> lossFunc = {
      CCE, CCE, CCE, CCE, CCE, CCE, // Hidden layers
      MSE
    };
};

#endif // NEURAL_CONFIG_H