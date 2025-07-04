//Project: NEURA 
//NEURA is the artificial neuron and network development research for the artificial intellegence for data analysis
//Searching a right structure of the neural network and optimize an artificial fast neuron in most optimal codes
//After the further development and edges, the Project Mei as neuron investigation project and Project Raiden as network
//After Project NEURA: Artificial Neuron Network is execute then escalate into a hardware mode of neuron architecture
//Developer: Christopher Emmanuelle J. Visperas, Applied Physicist

#ifndef BOLTZMANN_H
#define BOLTZMANN_H

#include <iostream>
#include <vector>
#include <algorithm>
#include "layer.h"
#include "activation.h"

using namespace std;

class Boltzmann{
    private:
        vector <Layer> layer; // vector array of neural layer
        vector <size_t> num_neuron; // number of neurons in each layer
        vector <double> input, target;

        // weight between two interacting layer
        // i for the id of each layer, and j and k
        vector <vector<vector<double>>> weight;
        
        double learning_rate;
        ActivationFunction actFunc;

        double randomInRange() {
            static thread_local random_device rd; // Non-deterministic random device (if available)
            static thread_local mt19937 generator(rd()); // Mersenne Twister RNG seeded by random device
            uniform_real_distribution<double> distribution(-1, 1); // Uniform distribution between min and max
            return distribution(generator);
        }

        int randomState(){
            static thread_local random_device rd;
            static thread_local mt19937 generator(rd());
            uniform_int_distribution<int> distribution(-1, 1);
            return distribution(generator);
        }

        int normalize(vector<double>& data){
            auto x_min = min_element(data.begin(), data.end()), 
            x_max = max_element(data.begin(), data.end());

            int num = 0;

            for(size_t i = 0; i < data.size(); ++i){
                double x_normalized = (data[i] - double(*x_min)) / double(*x_max - *x_min);
                
                if(x_normalized > 0.5){
                    num = 1;
                }
                else if(x_normalized < 0.5){
                    num = -1;
                }
                else if (x_normalized == 0.5){
                    num = 0;
                }
            }

            return num;
        }

        public:
            Boltzmann(vector<size_t> num_neuron, const vector<double>& inputs,  const vector<double>& targets, double learningRate, ActivationFunction actfunc)
            : num_neuron(num_neuron), input(inputs), target(targets), learning_rate(learningRate), actFunc(actfunc){

                srand(time(0));
                layer.resize(num_neuron.size());

                weight.resize(num_neuron.size());
                
                for(size_t i = 0; i < num_neuron.size(); ++i){
                    weight[i].resize(num_neuron[i]);

                    size_t total = num_neuron[i] + num_neuron[i + 1];

                    for(size_t j = 0; j < num_neuron[i]; ++j){
                        weight[i][j].resize(num_neuron[i]);

                        for(size_t k = 0; k < num_neuron[i + 1]; ++k){
                            if(j == k){
                                weight[i][j][k] = randomInRange();
                            }
                            else{

                            }
                        }
                    }
                }
            }
};

#endif