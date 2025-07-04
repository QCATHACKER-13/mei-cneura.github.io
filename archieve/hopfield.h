//Project: NEURA 
//NEURA is the artificial neuron and network development research for the artificial intellegence for data analysis
//Searching a right structure of the neural network and optimize an artificial fast neuron in most optimal codes
//After the further development and edges, the Project Mei as neuron investigation project and Project Raiden as network
//After Project NEURA: Artificial Neuron Network is execute then escalate into a hardware mode of neuron architecture
//Developer: Christopher Emmanuelle J. Visperas, Applied Physicist

#ifndef HOPFIELD_H
#define HOPFIELD_H

#include <iostream>
#include <vector>
#include <algorithm>
#include "cneuron.h"
#include "activation.h"
#include "layer.h"

using namespace std;

class Hopfield{
    private:
        vector <Neuron> neuron;
        vector<vector<double>> weight;
        vector<int> state;
        vector<double> bias, input, output, energy, target, error;
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
        Hopfield(size_t num_neuron, const vector<double>& inputs,  const vector<double>& targets, double learningRate, ActivationFunction actfunc)
        : input(inputs), target(targets), learning_rate(learningRate), actFunc(actfunc){

            weight.resize(num_neuron);
            bias.resize(num_neuron);
            state.resize(num_neuron);
            output.resize(num_neuron);
            error.resize(num_neuron);
            energy.resize(num_neuron);

            srand(time(0));

            for(size_t i = 0; i < num_neuron; ++i){
                neuron.emplace_back(Neuron(inputs, targets[i], learningRate, actFunc));
            }

            for(size_t i = 0; i < num_neuron; ++i){
                weight[i].resize(num_neuron);
                for(size_t j = 0; j < num_neuron; ++j){
                    if(i == j){
                        weight[i][j] = 0;
                    }
                    else {
                        weight[i][j] = randomInRange();
                    }
                }
                bias[i] = neuron[i].get_bias();
            }
        }

        void entropy(){
            for(size_t i = 0; i < neuron.size(); ++i){
                neuron[i].feedforward();
                output[i] = neuron[i].get_output();
                error[i] = neuron[i].get_error();
            }

            for(size_t i = 0; i < neuron.size(); ++i){
                state[i] = normalize(output);
            }

            for(size_t i = 0; i < neuron.size(); ++i){
                energy[i] = 0; //Initialize energy
                for(size_t j = 0; j < neuron.size(); ++j){
                    energy[i] += (weight[i][j] * double(state[i] * state[j]));
                }
                energy[i] += (bias[i] * double(state[i]));
            }
        }

        void enthalpy(){
            for(size_t i = 0; i < neuron.size(); ++i){
                error[i] = 0.5*(error[i] + (target[i] - energy[i]));
                state[i] = normalize(energy);
            }

            for(size_t i = 0; i < neuron.size(); ++i){
                neuron[i].backward();
            }

            for(size_t i = 0; i < neuron.size(); ++i){
                for(size_t j = 0; j < neuron.size(); ++j){
                    if(i == j){
                        weight[i][j] = 0;
                    }
                    else{
                        weight[i][j] = (learning_rate * error[i] * double(state[j]));
                    }
                }
                bias[i] = 0.5*(bias[i] + neuron[i].get_bias());
            }
        }

        void measurement(){
            for(size_t i = 0; i < neuron.size(); ++i){
                cout << i << " " << setw(7) << output[i] << " " << setw(7) << energy[i] << " " << setw(7) << state[i] << " ";;
                for(size_t j = 0; j < neuron.size(); ++j){
                    cout << setw(7) << weight[i][j] << " ";
                }
                cout << endl;
            }
            cout << endl;
        }

        vector<double> getEnergy(){
            return energy;
        }

        vector<int> getState(){
            return state;
        }

        vector <vector<double>> getWeight(){
            return weight;
        }

        vector<double> getBias(){
            return bias;
        }

        vector<double> getOutput(){
            return output;
        }

        vector<double> getError(){
            return error;
        }

};

#endif