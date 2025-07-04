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
#include <cmath>
#include <numeric>
#include "cneuron++.h"

using namespace std;

enum nltype{
    VISIBLE, HIDDEN, OUTPUT
};

class Layer{
    private:
    vector<Neuron> neuron;
    vector<double> input, bias, output, target, error, prediction, learning_rate, probability, loss_func;// Weights, inputs, output, bias
    vector<vector<double>> weight; // Weight storage for neurons
    
    actfunc actFunc; // Activation function type
    lrs lr_schedule; // Learning rate adjustment strategy
    optimizer opt; // Optimization algorithm

    void softmax() {
    //double maxVal = *max_element(output.begin(), output.end()); // Prevents overflow
            vector<double> exp_values(output.size());
        
            // Compute exponentials
            for (size_t i = 0; i < output.size(); ++i) {
                exp_values[i] = exp(output[i]);
            }
        
            double sum_exp = accumulate(exp_values.begin(), exp_values.end(), 0.0);
        
            probability.resize(output.size());
        
            // Compute softmax probabilities
            for (size_t i = 0; i < output.size(); ++i) {
                if (sum_exp != 0) {
                    probability[i] = exp_values[i] / sum_exp;
                } else if (sum_exp == 0) {
                    probability[i] = 0.0;
                }
            }
        }

        void cross_entropy_loss(const vector<double>& prob_target) {
            loss_func.resize(output.size());
            for (size_t i = 0; i < output.size(); i++){
                if (actFunc == SIGMOID || actFunc == TANH) {
                    loss_func[i] = prob_target[i] * log(output[i]) + (1 - prob_target[i]) * log(1 - prob_target[i]); // Binary Cross-Entropy
                } 
                else if (actFunc != SIGMOID || actFunc != TANH){
                    loss_func[i] = -target[i] * log(output[i]); // Cross-Entropy Loss for Softmax
                }
            }
        }

    public:
        Layer (size_t num_neuron, int step_size, 
            const vector<double>& inputs, const vector<double>& biasRange, const vector<double>& weightRange, const vector<double>& targets, 
            const vector<double>& momentum_learningRate, const vector<double> learning_rate, double decay_rate,
            nltype layertype, actfunc actfunc, lrs lr_schedule, optimizer opt)
        
        : input(inputs), target(targets), learning_rate(learning_rate), 
        actFunc(actfunc), lr_schedule(lr_schedule), opt(opt)
        
        {
            weight.resize(num_neuron);
            bias.resize(num_neuron);
            output.resize(num_neuron);
            error.resize(num_neuron);
            
            for(size_t i = 0; i < num_neuron; ++i){
                neuron.emplace_back(
                    Neuron(
                        step_size, inputs, biasRange, weightRange, targets[i], 
                        momentum_learningRate, learning_rate, decay_rate, 
                        actfunc, lr_schedule, opt
                    )
                );
            }
            
            for(size_t i = 0; i < num_neuron; i++){
                weight[i].resize(input.size());
                
                for(size_t j = 0; j < input.size(); ++j){
                    weight[i][j] = neuron[i].get_weight()[j];
                }
                bias[i] = neuron[i].get_bias();
            }
        }

        void feedforward() {
            for(size_t i = 0; i < neuron.size(); i++){ 
                neuron[i].feedforward();
                output[i] = neuron[i].get_output();
                error[i] = neuron[i].get_error();
                
                // Sync weights
                //weight[i] = neuron[i].get_weight();
            }
        }
        

        void backpropagation(){
            for(size_t i = 0; i < neuron.size(); i++){
                neuron[i].backward();
                //weight[i] = neuron[i].get_weight(); // Sync updates
            }
        }

        // Softmax function for probability distribution output
        void probability() {
            softmax();
            cross_entropy_loss();
        }

        void measurement(){
            /*for(size_t i = 0; i < neuron.size(); ++i){
                cout << i << " " << setw(7) << output[i] << " ";
                for(size_t j = 0; j < input.size(); ++j){
                    cout << setw(7) << weight[i][j] << " ";
                }
                cout << " " << setw(7) << bias[i] << endl;
            }*/
            for(size_t i = 0; i < neuron.size(); i++){
                neuron[i].print_neuron(i);
            }
            cout << endl;            
        }

        void training(int num_epochs, double error_margin, bool switcher){
            for(size_t i = 0; i < neuron.size(); i++){
                neuron[i].training(i, num_epochs, error_margin, switcher);
                
            }
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

        vector<Neuron>& getNeuron() {
            return neuron;
        }        

};

#endif