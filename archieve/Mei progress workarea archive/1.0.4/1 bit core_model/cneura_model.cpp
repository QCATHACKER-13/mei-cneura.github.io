#include<iostream>
#include<vector>
#include "../build_setting/net_builder.h"
#include "../config_setting/neuralnet_config.h"

int main(){
    //Define the number of neuron on each layer
    vector <size_t> num_neurons (3, 9);//(2, 2); //= {3, 3, 3, 3};
    // Define input and target output
    vector<double> inputs =  {5, 2, 4, 4, 1, 3, 8, 6, 7}; // Example target output
    vector<double> targets = {1, 2, 3, 4, 5, 6, 7, 8, 9};//{0.5, 0.1, 0.4}; // Example single input
    vector <double> label =  {0, 0, 0, 0, 1, 0, 0, 0, 0};
    vector <double> keep_prob = {0.5, 0.5, 0.5}; // Dropout rates for each layer
    vector<double> normalized_inputs = Data(inputs).dataset_normalization(SYMMETRIC); // Example single input
    vector<double> normalized_targets = Data(inputs).targetdataset_normalization(SYMMETRIC, targets); // Example single target output
    
    // Define hyperparameters
    double learning_rate = 1e-2;
    double decay_rate = 1e-6;
    vector<double> beta = {0.9, 0.9999}; // Momentum factor for Adam
    int step_size = 1000; // For step decay learning rate
    
    NeuralConfig config;
config.layers = {
    {9, TANH, 0.9},
    {9, LEAKY_RELU, 0.9},
    {9, LEAKY_RELU, 0.95}
};
config.learning_rate = 0.001;
config.step_size = 5000;

Neural net = NetBuilder::build(config, normalized_inputs);
net.set_input(normalized_inputs);
net.set_target(normalized_targets);
net.learning(config.step_size);

}