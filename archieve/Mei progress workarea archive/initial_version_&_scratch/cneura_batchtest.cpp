#include <iostream>
#include <vector>
#include "cneura.h"
#include "cdata_tools.h"

using namespace std;

int main() {
    // Define the number of neurons in each layer
    vector<size_t> num_neurons = {2, 4, 1}; // Example: 2 input neurons, 4 hidden neurons, 1 output neuron

    // Define training data (XOR problem)
    vector<vector<double>> train_X = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    vector<vector<double>> train_Y = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    // Define hyperparameters
    double learning_rate = 1e-2;
    double decay_rate = 1e-5;
    vector<double> beta = {0.9, 0.999}; // Momentum factors for Adam optimizer
    int step_size = 10000; // Number of steps for training
    int epochs = 1000;     // Number of epochs for batch training
    int batch_size = 2;    // Batch size for training

    // Initialize Neural Network with Adam optimizer and Step Decay learning rate
    Neural neural(
        num_neurons, 
        train_X[0], // Example input to initialize the network
        learning_rate, 
        decay_rate, 
        beta,
        LEAKY_RELU, // Activation function
        ITDECAY,    // Learning rate schedule
        ADAM,       // Optimizer
        MSE         // Loss function
    );

    // Train the network using the learning method
    cout << "Training using learning method..." << endl;

    // Train the network using batch training
    cout << "\nTraining using batch method..." << endl;
    neural.train_batch(train_X, train_Y, epochs, batch_size);

    // Print the results of each layer after training
    cout << "\nResults after training:" << endl;
    neural.print_layer_results();

    return 0;
}
