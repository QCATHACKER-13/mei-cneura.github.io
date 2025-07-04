//Project: NEURA 
//NEURA is the artificial neuron and network development research for the artificial intellegence for data analysis
//Searching a right structure of the neural network and optimize an artificial fast neuron in most optimal codes
//After the further development and edges, the Project Mei as neuron investigation project and Project Raiden as network
//After Project NEURA: Artificial Neuron Network is execute then escalate into a hardware mode of neuron architecture
//Developer: Christopher Emmanuelle J. Visperas, Applied Physicist

#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <vector>
#include <algorithm>
#include "cneuron.h"
#include "activation.h"

using namespace std;

class Layer{
    private:
        vector<Neuron> neuron;
        vector<vector<double>> weight;
        vector<double> input, bias, output, target, error;
        double learning_rate;
        ActivationFunction actFunc;

    public:
        Layer(size_t num_neuron, const vector<double>& inputs,  const vector<double>& targets, double learningRate, ActivationFunction actfunc)
            : input(inputs), target(targets), learning_rate(learningRate), actFunc(actfunc){

                weight.resize(num_neuron);
                bias.resize(num_neuron);
                output.resize(num_neuron);
                error.resize(num_neuron);

                for(size_t i = 0; i < num_neuron; ++i){
                    neuron.emplace_back(Neuron(inputs, targets[i], learningRate, actFunc));
                }
                
                for(size_t i = 0; i < num_neuron; ++i){
                    weight[i].resize(input.size());
                    
                    for(size_t j = 0; j < input.size(); ++j){
                        weight[i][j] = neuron[i].get_weight()[j];
                    }
                    bias[i] = neuron[i].get_bias();
                }
        }

        void feedforward(){
            for(size_t i = 0; i < neuron.size(); ++i){
                neuron[i].feedforward();
                output[i] = neuron[i].get_output();
                error[i] = neuron[i].get_error();
            }
        }

        void backpropagation(){
            for(size_t i = 0; i < neuron.size(); ++i){
                neuron[i].backward();
            }
        }

        void measurement(){
            for(size_t i = 0; i < neuron.size(); ++i){
                cout << i << " " << setw(7) << output[i] << " ";
                for(size_t j = 0; j < neuron.size(); ++j){
                    cout << setw(7) << weight[i][j] << " ";
                }
                cout << " " << setw(7) << bias[i] << endl;
            }
            cout << endl;
        }

        vector<double> getOutput(){
            return output;
        }

        vector <vector<double>> getWeight(){
            return weight;
        }

        vector<double> getBias(){
            return bias;
        }

        vector<double> getError(){
            return error;
        }

        vector<Neuron> getNeuron(){
            return neuron;
        }

};
#endif