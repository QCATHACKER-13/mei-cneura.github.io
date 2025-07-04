//Project: NEURA 
//NEURA is the artificial neuron and network development research for the artificial intellegence for data analysis
//Searching a right structure of the neural network and optimize an artificial fast neuron in most optimal codes
//After the further development and edges, the Project Mei as neuron investigation project and Project Raiden as network
//After Project NEURA: Artificial Neuron Network is execute then escalate into a hardware mode of neuron architecture
//Developer: Christopher Emmanuelle J. Visperas, Applied Physicist Graduated

#ifndef CNEURON_H
#define CNEURON_H

#include <cstdlib>   // for rand() and srand()
#include <ctime>     // for seeding rand()
#include <cmath>     // for exp()
#include <iostream>  // for debugging
#include <iomanip>  // for extension debugging
#include <random>  // For random number generation
#include <vector>
#include "cmatrix.h"
#include "activation.h"

using namespace std;

class Neuron{
    private:
        vector<double> weight, input;
        double target, bias, output, learning_rate, error, prediction;

        ActivationFunction actFunc;
        
        double randomInRange(double min, double max) {
            static thread_local random_device rd; // Non-deterministic random device (if available)
            static thread_local mt19937 generator(rd()); // Mersenne Twister RNG seeded by random device
            uniform_real_distribution<double> distribution(min, max); // Uniform distribution between min and max
            return distribution(generator);
        }

    public:
        Neuron(const vector<double>& inputs, double targets, double learningRate, ActivationFunction actfunc)
        : input(inputs), target(targets), learning_rate(learningRate), actFunc(actfunc){
            weight.resize(inputs.size());
            //input = inputs;
            //target = targets;


            srand(time(0));

            for(size_t i = 0; i < inputs.size(); ++i){
                weight[i] = randomInRange(-1.0, 1.0);
            }

            bias = randomInRange(-1.0, 1.0);
        }

        void feedforward(){
            output = (dot_product(input, weight) + bias);
            error = target - output;
        }

        void activation(){
            Activation activation(actFunc);
            prediction = activation.activation_value(output);
        }

        void backward(){
            //Updating weights
            for(size_t i = 0; i < input.size(); ++i){
                weight[i] += (learning_rate * error * input[i]);
            }

            //Updating bias
            bias += (learning_rate * error);
        }

        void print_neuron() {
            /*cout << "----------------------------------------" << endl;
            cout << "| Output  | Bias    | Weights          |" << endl;
            cout << "----------------------------------------" << endl;
            cout << " " << setw(7) << output << "  " << setw(7) << bias << "  ";
            for (double w : weight) {
                cout << setw(7) << w << " ";
            }
            cout << endl;
            
            Print table header
            cout << left << setw(col1_width) << "OUTPUT"
            << setw(col2_width) << "BIAS"
            << setw(col3_width) << "WEIGHTS" 
            << endl;
            cout << string(45, '-') << endl; */ //Print separator
            int col1_width = 10, col2_width = 15, col3_width = 20;

            // Print table row
            cout << setw(col1_width) << output 
            << setw(col2_width) << bias;
            for (double w : weight) {
                cout << setw(col3_width) << w;
            }
            cout << setw(col3_width) << error << endl;

        }

        void training(bool switcher){
            switch(switcher){
                case true:
                    feedforward();
                    print_neuron();
                    backward();
                    break;
                case false:
                    feedforward();
                    backward();
                    break;
            }
        }

        double get_output(){
            return output;
        }

        double get_bias(){
            return bias;
        }

        vector<double> get_weight(){
            return weight;
        }

        double get_error(){
            return error;
        }

        void set_weight(vector<double> weights){
            if(weights.size() == weight.size()){
                weight = weights;
            }
        }

        void set_bias(double biases){
            bias = biases;
        }
};



#endif