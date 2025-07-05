

#ifndef MODEL_OPTION_H
#define MODEL_OPTION_H

#pragma once

constexpr double ALPHA = 1e-3;
constexpr double EPSILON = 1e-12;
constexpr double CLIP_GRAD = 1.0;

// Enum for layer types
enum NEURONTYPE { INPUT, HIDDEN, OUTPUT }; // Enum for neuron types
enum ACTFUNC { SIGMOID, RELU, TANH, LEAKY_RELU, ELU}; // Enum for activation functions
enum OPTIMIZER { SGD, ADAGRAD, RMSPROP, ADAM}; // Enum for optimization algorithms
enum LEARNRATE { CONSTANT, STEPDECAY, EXPDECAY, ITDECAY}; // Enum for learning rate adjustment strategies
enum LOSSFUNC { MSE, MAE, BCE, CCE, HUBER}; // Enum for loss functions
enum DISTRIBUTION { NORMAL, UNIFORM }; // Enum for normalization types
enum INITIALIZATION {HE, XAVIER};

#endif