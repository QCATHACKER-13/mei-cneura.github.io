/* Project: NEURA 
NEURA is an artificial neuron and network development research project 
focused on artificial intelligence for data analysis. The goal is to find 
the optimal neural network structure and improve artificial neurons for 
efficiency and speed.

Project Development and Innovations:
- Project Neura: Neuron investigation project
- Project Mei: Neural Net Architecture investigation project

Developer: Christopher Emmanuelle J. Visperas, Applied Physics Researcher*/

#include <iostream>
#include <vector>
#include <chrono>
#include "cneura.h"
#include "../data_tools/cdata_tools.h"

using namespace std;

int main(){
    //Define the number of neuron on each layer
    vector <size_t> num_neurons (5, 9);//(2, 2); //= {3, 3, 3, 3};
    // Define input and target output
    vector<double> inputs =  {5, 2, 4, 4, 1, 3, 8, 6, 7}; // Example target output
    vector<double> targets = {1, 2, 3, 4, 5, 6, 7, 8, 9};//{0.5, 0.1, 0.4}; // Example single input
    vector <double> label =  {0, 0, 0, 0, 1, 0, 0, 0, 0};
    vector <double> keep_prob = {0.2, 0.3, 0.4, 0.5, 0.95}; // Dropout rates for each layer
    vector<double> normalized_inputs = Data(targets).targetdataset_normalization(SYMMETRIC, inputs); // Example single input
    vector<double> normalized_targets = Data(targets).dataset_normalization(SYMMETRIC); // Example single target output
    
    // Define hyperparameters
    double learning_rate = 1e-2;
    double decay_rate = 1e-6;
    vector<double> beta = {0.9, 0.9999}; // Momentum factor for Adam
    int step_size = 1000; // For step decay learning rate
    vector<ACTFUNC> actFunc = {
      LEAKY_RELU, LEAKY_RELU, LEAKY_RELU, LEAKY_RELU,
      LEAKY_RELU
    };

    // Initialize Neural Network with Adam optimizer and Step Decay learning rate
    Neural neural(
        num_neurons, Data(inputs).dataset_normalization(SYMMETRIC),
        learning_rate, decay_rate, beta,
        XAVIER, UNIFORM, actFunc, ITDECAY, ADAM, MSE
    );
    
    // Define hyperparameters
    int epochs = 1000;
    int batch_size = 1000;

    neural.set_step_size(step_size);
    neural.set_dropout(keep_prob); // Set dropout rate for regularization
    neural.set_input(normalized_inputs);
    neural.set_target(normalized_targets);
    //neural.set_softlabeling(label, 1e-6);
    neural.set_hardlabeling(label);
    
    // Train the network
    //neural.learning(step_size);
    neural.enforce_learning(step_size, 0.05);

    return 0;
}