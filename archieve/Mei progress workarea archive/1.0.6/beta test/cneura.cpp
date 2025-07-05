/* Project: NEURA 
NEURA is an artificial neuron and network development research project 
focused on artificial intelligence for data analysis. The goal is to find 
the optimal neural network structure and improve artificial neurons for 
efficiency and speed.

Future projects:
- Project Mei: Neuron investigation project

Developer: Christopher Emmanuelle J. Visperas, Applied Physics Researcher*/

#include <iostream>
#include <vector>
#include <chrono>
#include "cneura.h"
#include "../data_tools/cdata_tools.h"
//#include "../model_options/model_option.h"

using namespace std;

int main(){
    //Define the number of neuron on each layer
    vector <size_t> num_neurons (5, 9);//(2, 2); //= {3, 3, 3, 3};
    // Define input and target output
    vector<double> inputs =  {5, 2, 4, 4, 1, 3, 8, 6, 7}; // Example target output
    vector<double> targets = {1, 2, 3, 4, 5, 6, 7, 8, 9};//{0.5, 0.1, 0.4}; // Example single input
    vector <double> label =  {0, 0, 0, 0, 0, 0, 0, 0, 1};
    vector <double> keep_prob = {0.2, 0.3, 0.4, 0.5, 0.95}; // Dropout rates for each layer
    vector<double> normalized_inputs = Data(targets).targetdataset_normalization(SYMMETRIC, inputs); // Example single input
    vector<double> normalized_targets = Data(targets).dataset_normalization(SYMMETRIC); // Example single target output
    
    // Define hyperparameters
    double learning_rate = 1e-1;
    double decay_rate = 1e-6;
    vector<double> beta = {0.9, 0.9999}; // Momentum factor for Adam
    int step_size = 1000; // For step decay learning rate
    vector<ACTFUNC> actFunc = {
      TANH, TANH, TANH, TANH,
      LEAKY_RELU
    };
    vector<LOSSFUNC> lossFunc = {
      CCE, CCE, CCE, CCE,
      MSE
    };

    // Initialize Neural Network with Adam optimizer and Step Decay learning rate
    Neural neural(
        num_neurons, Data(inputs).dataset_normalization(SYMMETRIC),
        learning_rate, decay_rate, beta,
        XAVIER, UNIFORM, actFunc, ITDECAY, ADAM, lossFunc
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
    //neural.gaussianlabelling(label, 5, 1e-6); // Set soft labeling for the output layer
    
    // Train the network
    neural.set_task(REGRESSION); // Set task type to classification
    //neural.enforce_learning(step_size, 0.05);
    neural.learning(step_size, 0.05);
    neural.set_task(CLASSIFICATION); // Set task type to classification
    //neural.enforce_learning(step_size, 0.05);
    neural.learning(step_size, 0.05);

    return 0;
}