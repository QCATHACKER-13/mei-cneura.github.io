#include <iostream>
#include <vector>
#include "clayer.h"

using namespace std;

int main() {
    // Define input and target output
    vector<double> inputs = {1, 6, 9}; // Example single input
    vector<double> target = {2, 5, 8}; // Expected outputs for each neuron
    vector<double> labeling = {0, 1, 0}; // Example labeling for softmax output

    double learning_rate = 1e-2;
    double decay_rate = 0.001;
    vector<double> beta = {0.9, 0.9999}; // Momentum factor for Adam
    int step_size = 10; // For step decay learning rate

    // Initialize Layer with Adam optimizer and Step Decay learning rate
    Layer h1layer(
        3, inputs, 
        learning_rate, decay_rate, beta,
        HIDDEN, LEAKY_RELU, EXPDECAY, ADAM, MSE
    );

    //Perform feedforward operation
    h1layer.set_step_size(step_size);
    h1layer.normalization(MEANCENTER);
    h1layer.feedforward();

    Layer h2layer(
        3, h1layer.get_activated_output(), // Ensure input matches the output of the previous layer
        learning_rate, decay_rate, beta,
        HIDDEN, LEAKY_RELU, EXPDECAY, ADAM, MSE
    );

    h2layer.set_step_size(step_size);
    h2layer.feedforward();
    
    Layer olayer(
        3, h2layer.get_output(), // Ensure input matches the output of the previous layer
        learning_rate, decay_rate, beta,
        OUTPUT, LEAKY_RELU, EXPDECAY, ADAM, MSE
    );

    olayer.set_step_size(step_size);
    olayer.set_target(target);
    olayer.set_labeling(labeling, 1e-6);
    olayer.feedforward();

    olayer.probability_calculation();
    h2layer.set_error(olayer.get_error());
    h2layer.probability_calculation();
    h1layer.set_error(olayer.get_error());
    h1layer.probability_calculation();

    h1layer.debug_state();
    h2layer.debug_state();
    olayer.debug_state();

    // Perform backpropagation
    olayer.backpropagation();
    h2layer.backpropagation();
    h1layer.backpropagation();

    //layer.training(1000, 0.01, true);

    return 0;
}