/* Project: NEURA 
NEURA is an artificial neuron and network development research project 
focused on artificial intelligence for data analysis. The goal is to find 
the optimal neural network structure and improve artificial neurons for 
efficiency and speed.

Project Development and Innovations:
- Project Neura: Neuron investigation project
- Project Mei: Neural Net Architecture investigation project

Developer: Christopher Emmanuelle J. Visperas, Applied Physics Researcher*/


#ifndef CLAYER_H
#define CLAYER_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <numeric>   // for accumulate
#include "cneuron.h"

using namespace std;

// Enum for layer types
//enum nltype { DEFAULT, HIDDEN, OUTPUT };

class Layer{
    private:
        // Layer properties
        int step_size, timestep = 0; // Training step parameters
        double decay_rate, loss = 0.0; // Decay rate for learning rate, loss function
        neurontype layertype; // Layer type (input, hidden, output)
        
        // Layer parameters
        vector<Neuron> neuron;
        vector<double> input, bias, gradient_bias, output, activated_output, 
            target, probability_target, error,
            learning_rate, beta, probability;
        vector<vector<double>> weight, gradient_weight; // Weight storage for neurons
        
        actfunc actFunc; // Activation function type
        lrs lr_schedule; // Learning rate adjustment strategy
        optimizer opt; // Optimization algorithm
        lossfunc lossFunc; // Loss function type
        neurontype ntype; // Neuron type (input, hidden, output)

    public:
        // Constructor for initializing neural layer properties
        Layer(size_t num_neuron,
            const vector<double>& inputs,
            const double& learningRate, const double& decay_rate, const vector<double>& beta,
            neurontype ntype, actfunc actfunc, lrs lr_schedule, optimizer opt, lossfunc lossFunc)
          : input(inputs),
            decay_rate(decay_rate), beta(beta),
            ntype(ntype), actFunc(actfunc), lr_schedule(lr_schedule), opt(opt), lossFunc(lossFunc) {
                
                // Validate input size
                if (inputs.empty()) {
                    throw invalid_argument("Input vector cannot be empty.");
                }
                
                // Initialize layer properties
                weight.resize(num_neuron);
                bias.resize(num_neuron);
                output.resize(num_neuron);
                activated_output.resize(num_neuron);
                error.resize(num_neuron, 1.0);
                learning_rate.resize(num_neuron);
                gradient_weight.resize(num_neuron);
                gradient_bias.resize(num_neuron);
                probability.resize(num_neuron);
                probability_target.resize(num_neuron);
                
                for (size_t i = 0; i < num_neuron; ++i) {
                    neuron.emplace_back(
                        Neuron(
                            inputs,
                            learningRate, decay_rate, beta,
                            ntype, actfunc, lr_schedule, opt, lossFunc
                        )
                    );
                }
                
                // Initialize weights and biases for each neuron
                for (size_t i = 0; i < num_neuron; ++i) {
                    neuron[i].initialize();
                }
                
                // Initialize weights and biases for the layer
                for (size_t i = 0; i < num_neuron; i++) {
                    weight[i].resize(input.size());
                    gradient_weight[i].resize(input.size());
                    for (size_t j = 0; j < input.size(); ++j) {
                        weight[i][j] = neuron[i].get_weight()[j];
                    }
                    bias[i] = neuron[i].get_bias();
                }
            }

            void feedforward() {
                if (input.empty()) {
                    throw runtime_error("Input vector is empty in Layer::feedforward().");
                }
            
                for (size_t i = 0; i < neuron.size(); i++) {
                    neuron[i].feedforward();
            
                    output[i] = neuron[i].get_output();
                    activated_output[i] = neuron[i].get_activated_output();
                    bias[i] = neuron[i].get_bias();
            
                    for (size_t j = 0; j < input.size(); ++j) {
                        weight[i][j] = neuron[i].get_weight()[j];
                    }
            
                    learning_rate[i] = neuron[i].get_learning_rate();
                }
            }

        void softmax() {
            assert(ntype == OUTPUT); // Ensure layer is output layer
        
            double maxVal = *max_element(activated_output.begin(), activated_output.end()); // Prevent overflow

            vector<double> exp_values(activated_output.size());
        
            // Compute exponentials and sum
            double sum_exp = 0.0;
            for (size_t i = 0; i < activated_output.size(); ++i) {
                exp_values[i] = exp(activated_output[i] - maxVal); // Subtract max for numerical stability
                sum_exp += exp_values[i];
            }

            assert(sum_exp > 0); // Ensure sum_exp is not zero
            for (size_t i = 0; i < activated_output.size(); ++i) {
                probability[i] = exp_values[i] / sum_exp;
            }
        }

        // Loss functions and their derivatives
        void loss_function() {
            assert(ntype == OUTPUT); // Ensure layer is output layer
        
            double delta = 1.0; // Huber loss delta
            double sum_loss = 0.0;
            for (size_t i = 0; i < neuron.size(); ++i) {
        
                switch (lossFunc) {
                    case MSE:
                        sum_loss += 0.5 * pow(probability_target[i] - probability[i], 2); // Mean squared error
                        break;
        
                    case BCE:
                        sum_loss += -(probability_target[i] * log(probability[i]) +
                                  (1 - probability_target[i]) * log(1 - probability[i])) / probability.size();
                        break;
        
                    case CCE:
                        sum_loss += -(probability_target[i] * log(probability[i])) / activated_output.size();
                        break;
        
                    case HUBER:
                        if (abs(probability_target[i] - probability[i]) <= delta) {
                            sum_loss += 0.5 * pow(probability_target[i] - probability[i], 2); // Quadratic region
                        } else {
                            sum_loss += delta * (abs(probability_target[i] - probability[i]) - 0.5 * delta); // Linear region
                        }
                        break;
                }
            }
            loss = sum_loss / neuron.size(); // Average loss
        }

        //To calculate the error between output and target
        void loss_derivative() {
            assert(ntype == OUTPUT); // Ensure layer is output layer
        
            double delta = 1.0; // Huber loss delta
            for (size_t i = 0; i < neuron.size(); ++i) {
                double diff = probability_target[i] - probability[i];
        
                switch (lossFunc) {
                    case MSE:
                        error[i] = diff; // Mean squared error
                        break;
        
                    case BCE:
                        error[i] = -(probability_target[i] / probability[i]) +
                                   ((1 - probability_target[i]) / (1 - probability[i])); // Binary cross entropy
                        break;
        
                    case CCE:
                        error[i] = -(probability_target[i] / probability[i]); // Categorical cross entropy
                        break;
        
                    case HUBER:
                        if (abs(diff) <= delta) {
                            error[i] = diff; // Quadratic region
                        } else {
                            error[i] = delta * (diff > 0 ? 1 : -1); // Linear region
                        }
                        break;
                }
            }
        }
        

        void backpropagation(){
            for(size_t i = 0; i < neuron.size(); i++){
                neuron[i].backward();           
                for(size_t j = 0; j < input.size(); ++j){
                    gradient_weight[i][j] = neuron[i].get_gradient_weight()[j];
                    weight[i][j] = neuron[i].get_weight()[j];
                }
                gradient_bias[i] = neuron[i].get_gradient_bias();
                bias[i] = neuron[i].get_bias();
            }
        }

        void debug_state() const {
            cout << "Layer State:" << endl;
            cout << "Inputs: ";
            for (const auto& val : input) {
                cout << val << " ";
            }
            cout << endl;
        
            cout << "Outputs: ";
            for (const auto& val : output) {
                cout << val << " ";
            }
            cout << endl;
        
            cout << "Activated Outputs: ";
            for (const auto& val : activated_output) {
                cout << val << " ";
            }
            cout << endl;
        
            cout << "Biases: ";
            for (const auto& val : bias) {
                cout << val << " ";
            }
            cout << endl;
        
            cout << "Weights: " << endl;
            for (const auto& row : weight) {
                for (const auto& val : row) {
                    cout << val << " ";
                }
                cout << endl;
            }
            cout << endl;
        }

        vector <double> get_input() {return input;}
        vector<double> get_output(){return output;}
        vector<double> get_activated_output(){return activated_output;}
        vector <vector<double>> get_weight(){return weight;}
        vector<double> get_bias(){return bias;}
        double get_loss(){return loss;}
        vector<double> get_error(){return error;}
        vector<double> get_probability(){return probability;}
        vector<Neuron>& get_neuron() {return neuron;}
        
        void set_input(const vector<double>& inputs){
            assert(inputs.size() == input.size()); // Ensure sizes match
            this -> input = inputs;
            for(size_t i = 0; i < neuron.size(); i++){
                neuron[i].set_input(inputs);
            }
        }

        void set_bias(const vector<double>& biases){
            for(size_t i = 0; i < neuron.size(); i++){
                neuron[i].set_bias(biases[i]);
            }
        }
        void set_weight(const vector<vector<double>>& weights){
            for(size_t i = 0; i < neuron.size(); i++){
                neuron[i].set_weight(weights[i]);
            }
        }

        void set_target(const vector<double>& targets){
            assert(targets.size() == neuron.size()); // Ensure sizes match
            this -> target = targets;
        }

        void set_softlabeling(const vector<double>& labels, double resolution) {
            assert(labels.size() == neuron.size()); // Ensure sizes match
            size_t num_classes = labels.size();
        
            // Case 1: If resolution == 0, assume input is already a soft label (e.g., softmax output)
            if (resolution == 0.0) {
                this -> probability_target = labels; // direct use
                return;
            }
        
            // Case 2: Apply label smoothing (assuming labels is one-hot)
            probability_target.resize(num_classes);
            double smooth_val = resolution / num_classes;
        
            // Apply smoothing
            for (size_t i = 0; i < num_classes; ++i) {
                if (labels[i] == 1.0) {
                    this -> probability_target[i] = 1.0 - resolution + smooth_val;
                } else {
                    this -> probability_target[i] = smooth_val;
                }
            }
        }

        void set_hardlabeling(const vector<double>& targets){
            assert(targets.size() == neuron.size()); // Ensure sizes match
            this -> probability_target = targets;
        }

        void set_probability (const vector<double>& probabilities){
            assert(probabilities.size() == neuron.size()); // Ensure sizes match
            this -> probability = probabilities;
        }
        

        void set_error(const vector<double>& errors){
            for(size_t i = 0; i < neuron.size(); i++){
                neuron[i].set_error(errors[i]);
            }
        }

        void set_step_size(int stepsize){
            for(size_t i = 0; i < neuron.size(); i++){
                neuron[i].set_step_size(stepsize);
            }
        }
};

#endif