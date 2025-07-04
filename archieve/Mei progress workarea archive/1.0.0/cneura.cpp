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
#include "cneura.h"
#include "cdata_tools.h"

using namespace std;

int main(){
    //Define the number of neuron on each layer
    vector <size_t> num_neurons (3, 9);//(2, 2); //= {3, 3, 3, 3};
    // Define input and target output
    vector<double> targets = {1, 2, 3, 4, 5, 6, 7, 8, 9};//{0.5, 0.1, 0.4}; // Example single input
    vector <double> label =  {0, 0, 0, 0, 0, 0, 0, 0, 1};
    vector<double> inputs =  {5, 2, 4, 4, 1, 3, 8, 6, 7}; // Example target output
    double learning_rate = 1e-2;
    double decay_rate = 1e-5;
    vector<double> beta = {0.9, 0.9999}; // Momentum factor for Adam
    int step_size = 10000; // For step decay learning rate

    // Initialize Neural Network with Adam optimizer and Step Decay learning rate
    Neural neural(
        num_neurons, Data(inputs).normalization(SYMMETRIC),
        learning_rate, decay_rate, beta,
        LEAKY_RELU, ITDECAY, ADAM, MSE
    );
    
    // Define hyperparameters
    int epochs = 1000;
    int batch_size = 1000;
    
    // Train the network
    neural.learning(step_size, Data(targets).normalization(SYMMETRIC), label, "hardlabeling");

    return 0;
}