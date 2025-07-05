/* Project: NEURA 
NEURA is an artificial neuron and network development research project 
focused on artificial intelligence for data analysis. The goal is to find 
the optimal neural network structure and improve artificial neurons for 
efficiency and speed.

Project Development and Innovations:
- Project Neura: Neuron investigation project
- Project Mei: Neural Net Architecture investigation project

Developer: Christopher Emmanuelle J. Visperas, Applied Physics Researcher

*/
/*
This is neural configuration header file for the CNEURA project.
This file contains the necessary configurations for the neural network parameters,
including activation functions, learning rate schedules, optimizers, and loss functions.
It is used to initialize the neural network and set up the training process.
This file is part of the NEURA project, which focuses on developing artificial neurons
and networks for data analysis and artificial intelligence applications.
It is designed to be used with the C++17 language standard version.
*/
#ifndef NEURAL_CONFIG_H
#define NEURAL_CONFIG_H
#pragma once
#include <vector>

using namespace std;

// Enum for layer types
enum NEURONTYPE { INPUT, HIDDEN, OUTPUT }; // Enum for neuron types
enum ACTFUNC { SIGMOID, RELU, TANH, LEAKY_RELU, ELU}; // Enum for activation functions
enum OPTIMIZER { SGD, ADAGRAD, RMSPROP, ADAM}; // Enum for optimization algorithms
enum LEARNRATE { CONSTANT, STEPDECAY, EXPDECAY, ITDECAY}; // Enum for learning rate adjustment strategies
enum LOSSFUNC { MSE, MAE, BCE, CCE, HUBER}; // Enum for loss functions
enum DISTRIBUTION { NORMAL, UNIFORM }; // Enum for normalization types

struct NeuralConfig{
    vector<ACTFUNC> actFunc{
        LEAKY_RELU
    };

    LEARNRATE lr_schedule = LEARNRATE::ITDECAY;
    DISTRIBUTION distribution = DISTRIBUTION::UNIFORM; // Distribution type for weight initialization
    OPTIMIZER opt = OPTIMIZER::ADAM; // Optimizer type
    LOSSFUNC lossFunc = LOSSFUNC::MSE; // Loss function type
};

#endif // NEURAL_CONFIG_H