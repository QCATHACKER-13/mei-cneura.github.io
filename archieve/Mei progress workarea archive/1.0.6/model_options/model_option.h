/* Project: NEURA 
NEURA is an artificial neuron and network development research project 
focused on artificial intelligence for data analysis. The goal is to find 
the optimal neural network structure and improve artificial neurons for 
efficiency and speed.

Project Development and Innovations:
- Project Neura: Neuron investigation project
- Project Mei: Neural Net Architecture investigation project

Developer: Christopher Emmanuelle J. Visperas, Applied Physics Researcher

Note in this neuron testing in this following 
    - beta or momentum factor is set to 0.9 and 0.9999 as default, and must be the constant parameter
    - if beta is not a constant parameter then the target data with in the 
      range of the input data cause sometimes are error and more moderate error if the
      target data is out of range.


*/

#ifndef MODEL_OPTION_H
#define MODEL_OPTION_H

#pragma once

#include <stdexcept>

constexpr double ALPHA = 1e-3;
constexpr double EPSILON = 1e-8;
constexpr double CLIP_GRAD = 1.0;

// Enum for layer types
enum NEURONTYPE { INPUT, HIDDEN, OUTPUT }; // Enum for neuron types
enum ACTFUNC { SIGMOID, RELU, TANH, LEAKY_RELU, ELU}; // Enum for activation functions
enum OPTIMIZER { SGD, ADAGRAD, RMSPROP, ADAM}; // Enum for optimization algorithms
enum LEARNRATE { CONSTANT, STEPDECAY, EXPDECAY, ITDECAY}; // Enum for learning rate adjustment strategies
enum LOSSFUNC { MSE, MAE, BCE, CCE, HUBER}; // Enum for loss functions
enum DISTRIBUTION { NORMAL, UNIFORM }; // Enum for normalization types
enum INITIALIZATION {HE, XAVIER};
enum TASK { CLASSIFICATION, REGRESSION }; // Enum for task types
enum LABELLING { SOFT, HARD, GAUSSIAN }; // Enum for labeling types

#endif