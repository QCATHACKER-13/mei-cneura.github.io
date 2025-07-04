#include <iostream>
#include <vector>
#include <cassert>
#include "boltzmann.h"
#include "activation.h"

int main() {
    // Define input, target, learning rate, and activation function
    vector<double> inputs = {0.5, -0.3, 0.8};
    vector<double> targets = {0.6, -0.4, 0.7};
    double learningRate = 0.01;
    ActivationFunction actFunc = ActivationFunction::RELU; // Assuming SIGMOID is defined in activation.h
    
    Boltzmann bm(3, inputs, targets, learningRate, actFunc);

    bm.entropy();
    bm.enthalpy();
    bm.measurement();

    bm.entropy();
    bm.enthalpy();
    bm.measurement();

    return 0;
}