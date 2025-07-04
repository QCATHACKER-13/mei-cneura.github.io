#include <iostream>
#include <vector>
#include "layer++.h"

using namespace std;

int main() {
    // Define input and target output
    vector<double> inputs = {0.5, 0.1, 0.4}; // Example single input
    vector<vector <double>> targets = {{1.0, 0.0}, {0.0, 1.0}, {1.0}}; // Expected outputs for each neuron

    // Define weight and bias ranges
    vector<double> weightRange = {-1.0, 1.0};
    vector<double> biasRange = {-1.0, 1.0};

    // Define learning rate and momentum for Adam optimizer
    vector<double> momentum_learningRate = {0.9, 0.999};
    vector <double> learning_rate = {10, 0.01};
    double decay_rate = 0.001;
    int step_size = 10; // For step decay learning rate

    // Initialize Layer with Adam optimizer and Step Decay learning rate
    Layer vlayer(
        2, step_size, inputs, biasRange, weightRange, targets[0], 
        momentum_learningRate, learning_rate, decay_rate, 
        VISIBLE, LEAKY_RELU, EXPDECAY, ADAM
    );

    Layer hlayer(
        2, step_size, inputs, biasRange, weightRange, targets[1], 
        momentum_learningRate, learning_rate, decay_rate, 
        HIDDEN, LEAKY_RELU, EXPDECAY, ADAM
    );

    Layer olayer(
        1, step_size, inputs, biasRange, weightRange, targets[2], 
        momentum_learningRate, learning_rate, decay_rate, 
        OUTPUT, LEAKY_RELU, EXPDECAY, ADAM
    );

    //Perform feedforward operation
    vlayer.feedforward();
    vlayer.measurement();

    hlayer.feedforward();
    hlayer.measurement();

    olayer.feedforward();
    olayer.measurement();

    // Perform backpropagation
    vlayer.backpropagation();
    hlayer.backpropagation();
    olayer.backpropagation();

    //layer.training(1000, 0.01, true);

    return 0;
}