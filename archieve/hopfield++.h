//Project: NEURA 
//NEURA is the artificial neuron and network development research for the artificial intelligence for data analysis
//Searching a right structure of the neural network and optimize an artificial fast neuron in most optimal codes
//After the further development and edges, the Project Mei as neuron investigation project and Project Raiden as network
//After Project NEURA: Artificial Neuron Network is executed then escalate into a hardware mode of neuron architecture
//Developer: Christopher Emmanuelle J. Visperas, Applied Physicist

#ifndef HOPFIELD_H
#define HOPFIELD_H

#include <iostream>
#include <vector>
#include <algorithm>
#include "activation.h"
#include "layer.h"

using namespace std;

class Hopfield {
private:
    Layer neurallayer;
    vector<int> state;
    vector<vector<double>> weight;
    vector<double> input, output, energy, target, error;
    double learning_rate;
    ActivationFunction actFunc;

    double randomInRange() {
        static thread_local random_device rd; // Non-deterministic random device (if available)
        static thread_local mt19937 generator(rd()); // Mersenne Twister RNG seeded by random device
        uniform_real_distribution<double> distribution(-1, 1); // Uniform distribution between min and max
        return distribution(generator);
    }

    int randomState() {
        static thread_local random_device rd;
        static thread_local mt19937 generator(rd());
        uniform_int_distribution<int> distribution(-1, 1);
        return distribution(generator);
    }

    int normalize(vector<double>& data) {
        auto x_min = min_element(data.begin(), data.end());
        auto x_max = max_element(data.begin(), data.end());

        int num = 0;

        for (size_t i = 0; i < data.size(); ++i) {
            double x_normalized = (data[i] - double(*x_min)) / double(*x_max - *x_min);

            if (x_normalized > 0.5) {
                num = 1;
            } else if (x_normalized < 0.5) {
                num = -1;
            } else if (x_normalized == 0.5) {
                num = 0;
            }
        }
        return num;
    }
    
    public:
        Hopfield(size_t num_neuron, const vector<double>& inputs, const vector<double>& targets, double learningRate, ActivationFunction actfunc)
        : input(inputs), target(targets), learning_rate(learningRate), actFunc(actfunc), neurallayer(num_neuron, inputs, targets, learningRate, actfunc) {
            
            weight.resize(num_neuron);
            //bias.resize(num_neuron);
            state.resize(num_neuron);
            output.resize(num_neuron);
            error.resize(num_neuron);
            energy.resize(num_neuron);
            
            srand(time(0));
            
            for(size_t i = 0; i < num_neuron; ++i){
                weight[i].resize(num_neuron);
                
                for(size_t j = 0; j < num_neuron; ++j){
                    if(i == j){
                        weight[i][j] = 0;
                    } else {
                        weight[i][j] = randomInRange();
                    }
                }
                //bias[i] = neurallayer[i].getBias()[i];
            }
        }
        
        void entropy() {
            neurallayer.feedforward();
            
            for (size_t i = 0; i < neurallayer.getOutput().size(); ++i) {
                output[i] = neurallayer.getOutput()[i];
                error[i] = neurallayer.getError()[i];
            }
            
            for (size_t i = 0; i < neurallayer.getNeuron().size(); ++i) {
                state[i] = normalize(output);
            }
            
            for (size_t i = 0; i < neurallayer.getNeuron().size(); ++i) {
                energy[i] = 0; // Initialize energy
                for (size_t j = 0; j < neurallayer.getNeuron().size(); ++j) {
                    energy[i] += (weight[i][j] * double(state[i] * state[j]));
                }
                energy[i] += (neurallayer.getBias()[i] * double(state[i]));
            }
            
            for (size_t i = 0; i < neurallayer.getNeuron().size(); ++i) {
                error[i] = 0.5 * (error[i] + (target[i] - energy[i]));
            }
        }
        
        void enthalpy() {
            for (size_t i = 0; i < neurallayer.getNeuron().size(); ++i) {
                state[i] = normalize(energy);
            }
            
            neurallayer.backpropagation();
            
            for (size_t i = 0; i < neurallayer.getNeuron().size(); ++i) {
                for (size_t j = 0; j < neurallayer.getNeuron().size(); ++j) {
                    if (i == j) {
                        weight[i][j] = 0;
                    }
                    else {
                        weight[i][j] = (learning_rate * error[i] * double(state[j]));
                    }
                }
                neurallayer.getBias()[i] = (learning_rate * error[i]);
            }
        }
        
        void measurement() {
            for (size_t i = 0; i < neurallayer.getNeuron().size(); ++i) {
                cout << i << " " << setw(7) << output[i] << " " << setw(7) << energy[i] << " " << setw(7) << state[i] << " ";
                for (size_t j = 0; j < neurallayer.getNeuron().size(); ++j) {
                    cout << setw(7) << weight[i][j] << " ";
                }
                cout << endl;
            }
            cout << endl;
        }
        
        vector<double> getEnergy() {
            return energy;
        }
        
        vector<int> getState() {
            return state;
        }
        
        vector<vector<double>> getWeight() {
            return weight;
        }
        
        vector<double> getBias() {
            return neurallayer.getBias();
        }
        
        vector<double> getOutput() {
            return output;
        }
        
        vector<double> getError() {
            return error;
        }
};

#endif