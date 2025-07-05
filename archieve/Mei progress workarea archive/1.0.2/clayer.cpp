#include <iostream>
#include <vector>
#include "clayer.h"
#include "../data_tools/cdata_tools.h"

using namespace std;

int main() {
    // Define input and target output
    vector<double> inputs = {1, 6, 9}; // Example single input
    vector<double> target = {2, 5, 8}; // Expected outputs for each neuron
    vector<double> normalized_inputs = Data(inputs).dataset_normalization(SYMMETRIC); // Normalized input
    vector<double> normalized_targets = Data(inputs).targetdataset_normalization(SYMMETRIC, target); // Expected output
    vector<double> labeling = {0, 1, 0}; // Example labeling for softmax output

    double learning_rate = 1e-2;
    double decay_rate = 0.001;
    vector<double> beta = {0.9, 0.9999}; // Momentum factor for Adam
    int step_size = 10; // For step decay learning rate

    // Initialize Layer with Adam optimizer and Step Decay learning rate
    Layer h1layer(
        3, normalized_inputs, 
        learning_rate, decay_rate, beta,
        HIDDEN, LEAKY_RELU, EXPDECAY, ADAM, MSE
    );
    

    Layer h2layer(
        3, h1layer.get_activated_output(), // Ensure input matches the output of the previous layer
        learning_rate, decay_rate, beta,
        HIDDEN, LEAKY_RELU, EXPDECAY, ADAM, MSE
    );
    
    Layer olayer(
        3, h2layer.get_output(), // Ensure input matches the output of the previous layer
        learning_rate, decay_rate, beta,
        OUTPUT, LEAKY_RELU, EXPDECAY, ADAM, MSE
    );

    h1layer.set_step_size(step_size);
    h2layer.set_step_size(step_size);
    olayer.set_step_size(step_size);

    h1layer.set_dropout(0.8); // Set dropout rate for regularization
    h2layer.set_dropout(0.5); // Set dropout rate for regularization
    olayer.set_dropout(1.0); // Set dropout rate for regularization

    olayer.set_target(normalized_targets);
    olayer.set_softlabeling(labeling, 1e-6);

    h1layer.feedforward();
    h2layer.set_input(h1layer.get_activated_output());
    h2layer.feedforward();
    olayer.set_input(h2layer.get_activated_output());
    olayer.feedforward();
    olayer.softmax();
    
    h2layer.set_error(olayer.get_error());
    h1layer.set_error(olayer.get_error());

    h1layer.debug_state();
    h2layer.debug_state();
    olayer.debug_state();

    h1layer.regularization();
    h2layer.regularization();
    olayer.regularization();

    // Perform backpropagation
    olayer.backpropagation();
    h2layer.backpropagation();
    h1layer.backpropagation();

    h1layer.feedforward();
    h2layer.set_input(h1layer.get_activated_output());
    h2layer.feedforward();
    olayer.set_input(h2layer.get_activated_output());
    olayer.feedforward();
    olayer.softmax();

    h1layer.debug_state();
    h2layer.debug_state();
    olayer.debug_state();

    //layer.training(1000, 0.01, true);

    return 0;
}