#include <iostream>
#include <vector>
#include "cneuron++.h"

using namespace std;

int main() {
    // Define input and target output
    vector<double> inputs = {0.5, 0.1, 0.4}; // Example single input
    double target = 0.8; // Expected output

    // Define weight and bias ranges
    vector<double> weightRange = {-1.0, 1.0};
    vector<double> biasRange = {-0.5, 0.5};

    // Define learning rate and momentum for Adam optimizer
    vector<double> momentum_learningRate = {0.9, 0.999};
    vector <double> learning_rate = {1.0, 0.01};
    double decay_rate = 0.001;
    int step_size = 10; // For step decay learning rate

    // Initialize Neuron with Adam optimizer and Step Decay learning rate
    Neuron neuron(
        step_size, inputs, biasRange, weightRange, target, 
        momentum_learningRate, learning_rate, decay_rate, 
        LEAKY_RELU, EXPDECAY, ADAM
    );

    // Train the neuron for 1000 epochs with an error margin of 0.01
    neuron.training(1, 1000, 0.01, true);

    // Print final results
    cout << "\nFinal Weights: ";
    for (double w : neuron.get_weight()) {
        cout << w << " ";
    }
    cout << "\nFinal Bias: " << neuron.get_bias() << endl;
    cout << "Final Output: " << neuron.get_output() << endl;
    cout << "Final Error: " << neuron.get_error() << endl;

    return 0;
}
