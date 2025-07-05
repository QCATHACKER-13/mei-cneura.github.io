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


#ifndef CLAYER_H
#define CLAYER_H
#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <memory>   // for accumulate
#include "cneuron.h"

using namespace std;

// Enum for layer types
//enum nltype { DEFAULT, HIDDEN, OUTPUT };

class Layer{
    private:
        // Layer properties
        int timestep = 0; // Training step parameters
        double decay_rate, loss = 0.0; // Decay rate for learning rate, loss function
        
        // Layer parameters
        vector<unique_ptr<Neuron>> neurons;
        vector<double> input, bias, gradient_bias, output, activated_output, 
            target, probability_target, error,
            learning_rate, beta, probability;
        vector<vector<double>> weight, gradient_weight; // Weight storage for neurons
        
        ACTFUNC actFunc; // Activation function type
        LEARNRATE lr_schedule; // Learning rate adjustment strategy
        OPTIMIZER opt; // Optimization algorithm
        LOSSFUNC lossFunc; // Loss function type
        NEURONTYPE ntype; // Neuron type (input, hidden, output)

    public:
        // Constructor for initializing neural layer properties
        Layer(size_t num_neuron,
            vector<double> inputs,
            const double& learningRate, const double& decay_rate, const vector<double>& beta,
            NEURONTYPE ntype, ACTFUNC actfunc, LEARNRATE lr_schedule, OPTIMIZER opt, LOSSFUNC lossFunc)
          : input(inputs),
            decay_rate(decay_rate), beta(beta),
            ntype(ntype), actFunc(actfunc), lr_schedule(lr_schedule), opt(opt), lossFunc(lossFunc) {
                
                // Validate input size
                if (inputs.empty()){throw invalid_argument("Input vector cannot be empty.");}
                
                // Initialize layer properties
                neurons.reserve(num_neuron);
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
                    auto neuron = make_unique<Neuron>(
                    inputs,
                    learningRate, decay_rate, beta,
                    ntype, actFunc, lr_schedule, opt, lossFunc
                );
                
                if (!neuron) throw runtime_error("Failed to initialize layer " + to_string(i));
                neuron ->initialize(); // Initialize each neuron
                neurons.emplace_back(move(neuron));
                }
                if (neurons.empty()) {throw runtime_error("No layers were initialized in the neural network.");}

                // Initialize weights and biases for the layer
                for (size_t i = 0; i < num_neuron; i++) {
                    weight[i].resize(input.size());
                    gradient_weight[i].resize(input.size());
                    //this->weight[i] = neurons[i]->get_weight();
                    for (size_t j = 0; j < input.size(); ++j) {
                        this->weight[i][j] = neurons[i]->get_weight()[j];
                    }
                this->bias[i] = neurons[i]->get_bias();
            }
        }

        void initialize() {
            #pragma omp parallel for
            for (size_t i = 0; i < neurons.size(); ++i) {
                neurons[i]->initialize();
            }
        }
        
        void feedforward() {
            if (input.empty())throw runtime_error("Input vector is empty in Layer::feedforward().");
            
            #pragma omp parallel for
            for (size_t i = 0; i < neurons.size(); i++) {
                neurons[i]->feedforward();
                
                this -> output[i] = neurons[i] -> get_output();
                this -> activated_output[i] = neurons[i] -> get_activated_output();
                this -> bias[i] = neurons[i] -> get_bias();
                
                for (size_t j = 0; j < input.size(); ++j) {
                    this -> weight[i][j] = neurons[i] -> get_weight()[j];
                }
                
                this -> learning_rate[i] = neurons[i] -> get_learning_rate();
            }
        }

        void softmax() {
            assert(ntype == OUTPUT); // Ensure layer is output layer
        
            double maxVal = *max_element(activated_output.begin(), activated_output.end()); // Prevent overflow

            vector<double> exp_values(activated_output.size());
        
            // Compute exponentials and sum
            #pragma omp parallel for
            double sum_exp = 0.0;
            for (size_t i = 0; i < activated_output.size(); ++i) {
                exp_values[i] = exp(activated_output[i] - maxVal); // Subtract max for numerical stability
                sum_exp += exp_values[i];
            }

            assert(sum_exp > 0); // Ensure sum_exp is not zero
            for (size_t i = 0; i < activated_output.size(); ++i) {
                this -> probability[i] = exp_values[i] / sum_exp;
            }
        }

        // Loss functions and their derivatives
        void loss_function() {
            assert(ntype == OUTPUT); // Ensure layer is output layer
        
            double delta = 1.0; // Huber loss delta
            double sum_loss = 0.0;

            #pragma omp parallel for reduction(+:sum_loss)
            for (size_t i = 0; i < neurons.size(); ++i) {
        
                switch (lossFunc) {
                    case MSE:
                        sum_loss += 0.5 * pow(probability_target[i] - probability[i], 2); // Mean squared error
                        break;

                    case MAE:
                        sum_loss += abs(probability_target[i] - probability[i]); // Mean absolute error
                        break;
        
                    case BCE:
                        sum_loss += -(probability_target[i] * log(probability[i]) +
                                  (1 - probability_target[i]) * log(1 - probability[i]));
                        break;
        
                    case CCE:
                        sum_loss += -(probability_target[i] * log(probability[i]));
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
            this -> loss = sum_loss / neurons.size(); // Average loss
        }

        //To calculate the error between output and target
        void loss_derivative() {
            assert(ntype == OUTPUT); // Ensure layer is output layer
        
            double delta = 1.0; // Huber loss delta

            #pragma omp parallel for
            for (size_t i = 0; i < neurons.size(); ++i) {
                double diff = probability_target[i] - probability[i];
        
                switch (lossFunc) {
                    case MSE:
                        this -> error[i] = diff; // Mean squared error
                        break;

                    case MAE:
                        this -> error[i] = (diff > 0 ? 1 : -1); // Mean absolute error
                        break;
        
                    case BCE:
                        this -> error[i] = -(probability_target[i] / probability[i]) +
                                   ((1 - probability_target[i]) / (1 - probability[i])); // Binary cross entropy
                        break;
        
                    case CCE:
                        this -> error[i] = -(probability_target[i] / probability[i]); // Categorical cross entropy
                        break;
        
                    case HUBER:
                        if (abs(diff) <= delta) {
                            this -> error[i] = diff; // Quadratic region
                        } else {
                            this -> error[i] = delta * (diff > 0 ? 1 : -1); // Linear region
                        }
                        break;
                }
            }
        }

        void regularization() {
            int lambda_weight = 0, lambda_bias = 0;
            #pragma omp parallel for
            for(size_t i = 0; i < neurons.size(); i++){
                if(neurons[i] -> get_bias() < 0){
                    lambda_bias = -1;
                }
                else if(neurons[i] -> get_bias() > 0){
                    lambda_bias = 1;
                }
                loss += (lambda_bias * pow(neurons[i] -> get_bias(), 2))
                    + (lambda_bias * abs(neurons[i] -> get_bias()));

                for(size_t j = 0; j < input.size(); ++j){
                    if(neurons[i] -> get_weight()[j] < 0){
                        lambda_weight = -1;
                    }
                    else if(neurons[i] -> get_weight()[j] > 0){
                        lambda_weight = 1;
                    }
                    loss += (lambda_weight * pow(neurons[i] -> get_weight()[j], 2)) 
                    + (lambda_weight * abs(neurons[i] -> get_weight()[j]));
                }
                neurons[i] -> regularizated();
            }
        }
        

        void backpropagation(){
            #pragma omp parallel for
            for(size_t i = 0; i < neurons.size(); i++){
                neurons[i] -> backward();           
                for(size_t j = 0; j < input.size(); ++j){
                    this -> gradient_weight[i][j] = neurons[i] -> get_gradient_weight()[j];
                    this -> weight[i][j] = neurons[i] -> get_weight()[j];
                }
                this -> gradient_bias[i] = neurons[i] -> get_gradient_bias();
                this -> bias[i] = neurons[i] -> get_bias();
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
        
        // Getter functions for layer parameters
        vector<double> get_learning_rate() { return learning_rate; }
        vector<double> get_input() const { return input; }
        vector<double> get_target() const { return target; }
        vector<double> get_output() const { return output; }
        vector<double> get_activated_output() const { return activated_output; }
        vector<vector<double>> get_weight() const { return weight; }
        vector<double> get_bias() const { return bias; }
        double get_loss() const { return loss; }
        vector<double> get_error() const { return error; }
        vector<double> get_probability_target() const { return probability_target; }
        vector<double> get_probability() const { return probability; }
        const vector<unique_ptr<Neuron>>& get_neuron() const noexcept {return neurons;}
        NEURONTYPE get_neuron_type() const noexcept {return ntype;}

        void set_dropout(const double& keep_prob) {
            for (size_t i = 0; i < neurons.size(); ++i) {
                neurons[i]->set_dropout(keep_prob);
            }
        }

        void set_input(vector<double> inputs){
            assert(inputs.size() == input.size());
            this->input = inputs;
            for(size_t i = 0; i < neurons.size(); i++){
                neurons[i]->set_input(inputs);
            }
        }

        void set_bias(const vector<double>& biases){
            assert(biases.size() == neurons.size());
            for(size_t i = 0; i < neurons.size(); i++){
                neurons[i]->set_bias(biases[i]);
            }
        }
        
        void set_weight(const vector<vector<double>>& weights){
            assert(weights.size() == neurons.size());
            for(size_t i = 0; i < neurons.size(); i++){
                assert(weights[i].size() == input.size());
                neurons[i]->set_weight(weights[i]);
            }
        }
        
        void set_error(const vector<double>& errors){
            assert(errors.size() == neurons.size());
            for(size_t i = 0; i < neurons.size(); i++){
                neurons[i]->set_error(errors[i]);
            }
        }

        void set_target(const vector<double>& targets){
            assert(targets.size() == neurons.size()); // Ensure sizes match
            this->target = targets;
            
            for (size_t i = 0; i < neurons.size(); ++i) {
                neurons[i]->set_target(targets[i]);
            }
        }

        void set_softlabeling(const vector<double>& labels, double resolution) {
            assert(labels.size() == neurons.size()); // Ensure sizes match
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
            assert(targets.size() == neurons.size()); // Ensure sizes match
            this -> probability_target = targets;
        }

        void set_step_size(int step_size){
            for(size_t i = 0; i < neurons.size(); i++){
                neurons[i] -> set_step_size(step_size);
            }
        }
};

#endif