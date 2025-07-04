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
#include <iostream>
#include <vector>
#include "cneuron.h"
#include "../data_tools/cdata_tools.h"


using namespace std;

int main() {
    // Define input and target output
    vector<double> inputs = {1.0, 2.0, 3.0, 4.0}; // Example single input
    vector<double> normalized_inputs = Data(inputs).dataset_normalization(SYMMETRIC); // Normalized input
    double target = Data(inputs).targetdata_normalization(SYMMETRIC, 3.25); // Expected output

    // Define learning rate and momentum for Adam optimizer
    double learning_rate = 1e-2;
    vector<double> beta = {0.9, 0.9999}; // Momentum factor for Adam
    double decay_rate = 1e-6; // Decay rate for learning rate
    int step_size = 1000; // For step decay learning rate

    /*Note 
    the step size and total epochs for training are must be equal
    the target data must be within the range of the input data
    decay rate must be lower than the learning rate
    the learning rate must be lower than 1.0
    */


    // Initialize Neuron with Adam optimizer and Step Decay learning rate
    Neuron neuron(
        normalized_inputs, // Normalize input data
        learning_rate, decay_rate, beta,
        OUTPUT, LEAKY_RELU, ITDECAY, ADAM, MSE
    );

    neuron.set_step_size(step_size);
    neuron.set_target(target); // Set target value for training
    neuron.initialize(); // Initialize weights and bias
    // Train the neuron for 1000 epochs with an error margin of 0.01
    neuron.training(1, step_size, 0.05, true);

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
