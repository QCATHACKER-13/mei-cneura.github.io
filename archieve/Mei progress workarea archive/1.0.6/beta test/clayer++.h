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

#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <memory>   // for accumulate
#include <cassert>
#include "../core_model/cneuron.h"
#include "../data_tools/cdata_tools.h"
#include "../model_options/model_option.h"

using namespace std;

class Layer{
    private:
        // Layer properties
        size_t num_neurons; // Number of neurons in the layer
        int timestep = 0; // Training step parameters
        double decay_rate, loss = 0.0, avg_error; // Decay rate for learning rate, loss function
        
        // Layer parameters
        vector<unique_ptr<Neuron>> neurons;
        vector<double> input, bias, gradient, gradient_bias, 
            output, activated_output, activated_output_derivative,
            target, probability_target, error,
            learning_rate, beta, probability;
        vector<vector<double>> weight, gradient_weight; // Weight storage for neurons
        
        ACTFUNC actFunc; // Activation function type
        LEARNRATE lr_schedule; // Learning rate adjustment strategy
        OPTIMIZER opt; // Optimization algorithm
        LOSSFUNC lossFunc; // Loss function type
        NEURONTYPE ntype; // Neuron type (input, hidden, output)
        DISTRIBUTION distribution; // Distribution type for weight initialization
        INITIALIZATION initializer;
        TASK task; // Default task type
        LABELLING labelling; // Labeling type for output layer


    public:
        // Constructor for initializing neural layer properties
        Layer(size_t num_neuron,
            vector<double> inputs,
            const double& learningRate, const double& decay_rate, const vector<double>& beta,
            NEURONTYPE ntype, INITIALIZATION initializer, DISTRIBUTION distribution, 
            ACTFUNC actfunc, LEARNRATE lr_schedule, OPTIMIZER opt, LOSSFUNC lossFunc)
          : input(inputs),
            decay_rate(decay_rate), beta(beta),
            ntype(ntype), initializer(initializer), distribution(distribution), actFunc(actfunc), lr_schedule(lr_schedule), opt(opt), lossFunc(lossFunc) {
                
                // Validate input size
                if (inputs.empty()){throw invalid_argument("Input vector cannot be empty.");}
                
                // Initialize layer properties
                neurons.reserve(num_neuron);
                weight.resize(num_neuron);
                bias.resize(num_neuron);
                output.resize(num_neuron);
                activated_output.resize(num_neuron);
                activated_output_derivative.resize(num_neuron);
                target.resize(num_neuron);
                error.resize(num_neuron, 1.0);
                learning_rate.resize(num_neuron);
                gradient.resize(num_neuron);
                gradient_weight.resize(num_neuron);
                gradient_bias.resize(num_neuron);
                probability.resize(num_neuron);
                probability_target.resize(num_neuron);

                for (size_t i = 0; i < num_neuron; ++i) {
                    auto neuron = make_unique<Neuron>(
                    inputs,
                    learningRate, decay_rate, beta,
                    ntype, initializer, distribution, actFunc, lr_schedule, opt, lossFunc
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
                    for (size_t j = 0; j < input.size(); ++j) {
                        this->weight[i][j] = neurons[i]->get_weight()[j];
                    }
                this->bias[i] = neurons[i]->get_bias();
            }
        }

        void initialize() {
            #pragma omp parallel for
            for (size_t i = 0; i < neurons.size(); ++i) {
                //neurons[i]->initialize(neurons.size());
                neurons[i] -> initialization(neurons.size());
            }
        }
        
        void feedforward() {
            if (input.empty())throw runtime_error("Input vector is empty in Layer::feedforward().");
            
            #pragma omp parallel for
            for (size_t i = 0; i < neurons.size(); i++) {
                neurons[i]->feedforward();
                
                this -> output[i] = neurons[i] -> get_output();
                this -> activated_output[i] = neurons[i] -> get_activated_output();
                this -> activated_output_derivative[i] = neurons[i] -> get_derivative_activated_output();
                this -> bias[i] = neurons[i] -> get_bias();
                
                for (size_t j = 0; j < input.size(); ++j) {
                    this -> weight[i][j] = neurons[i] -> get_weight()[j];
                }
                
                this -> learning_rate[i] = neurons[i] -> get_learning_rate();
            }
        }

        void LayerNormalization() {
            double min = *min_element(activated_output.begin(), activated_output.end());
            double max = *max_element(activated_output.begin(), activated_output.end());
            
            double range = max - min;
            //assert(range > 0); // Ensure valid range

            double mean = accumulate(activated_output.begin(), activated_output.end(), 0.0) / activated_output.size();
            double stddev = sqrt(accumulate(activated_output.begin(), activated_output.end(), 0.0, 
                [mean](double sum, double val) { return sum + (val - mean) * (val - mean); }) / activated_output.size());
            //assert(stddev > 0); // Ensure valid standard deviation

            #pragma omp parallel for
            for(size_t i = 0; i  < activated_output.size(); i++){
                assert(i < activated_output.size()); // Ensure index is within bounds
                activated_output[i] = (activated_output[i] - mean) / stddev;
            }
        }

        void softmax() {
            assert(ntype == OUTPUT); // Ensure layer is output layer
            vector<double> exp_vals(activated_output.size(), 0.0);
            double maxVal = *max_element(activated_output.begin(), activated_output.end()); // Prevent overflow

            #pragma omp parallel for
            for (size_t i = 0; i < activated_output.size(); ++i) {
                exp_vals[i] = exp(activated_output[i] - maxVal);
            }
            double sum_exp = accumulate(exp_vals.begin(), exp_vals.end(), 0.0);
            for (size_t i = 0; i < activated_output.size(); ++i) {
                this->probability[i] = (sum_exp == 0) ? 0.0 : exp_vals[i] / sum_exp;
            }
        }

        // Improved regression_loss_function
        void regression_loss_function() {
            assert(ntype == OUTPUT); // Ensure layer is output layer
            double delta = 1.0; // Huber loss delta
            double sum_loss = 0.0;
            
            #pragma omp parallel for reduction(+:sum_loss)
            for (size_t i = 0; i < neurons.size(); ++i) {
                switch (lossFunc) {
                    case MSE:
                        sum_loss += 0.5 * pow(target[i] - activated_output[i], 2); // Mean squared error
                        break;
                        
                    case MAE:
                        sum_loss += abs(target[i] - activated_output[i]); // Mean absolute error
                        break;
                        
                    case BCE:
                        // Clamp activated_output for stability
                        sum_loss += -(target[i] * log(clamp(activated_output[i], EPSILON, 1.0 - EPSILON)) + 
                        (1 - target[i]) * log(1 - clamp(activated_output[i], EPSILON, 1.0 - EPSILON)));
                        break;
                    
                    case CCE:
                        // Clamp activated_output for stability
                        sum_loss += -(target[i] * log(max(EPSILON, activated_output[i])));
                        break;
                    
                    case HUBER:
                        if (abs(target[i] - activated_output[i]) <= delta) {
                            sum_loss += 0.5 * pow(target[i] - activated_output[i], 2); // Quadratic region
                        } else {
                            sum_loss += delta * (abs(target[i] - activated_output[i]) - 0.5 * delta); // Linear region
                        }
                        break;
                    }
                }
            this->loss += (sum_loss / neurons.size()); // Average loss
        }
            
        // Improved regression_loss_derivative
        void regression_loss_derivative() {
            assert(ntype == OUTPUT); // Ensure layer is output layer
            double delta = 1.0; // Huber loss delta
            
            #pragma omp parallel for
            for (size_t i = 0; i < neurons.size(); ++i) {
                switch (lossFunc) {
                    case MSE:
                        this->error[i] = (target[i] - activated_output[i]); // Mean squared error
                        break;
                    
                    case MAE:
                        this->error[i] = ((target[i] - activated_output[i]) > 0 ? 1.0 : -1.0); // Mean absolute error
                        break;
                    
                    
                    case BCE:
                        // Clamp activated_output for stability;
                        this->error[i] = -(target[i] / clamp(activated_output[i], EPSILON, 1.0 - EPSILON)) + 
                        ((1.0 - target[i]) / (1.0 - clamp(activated_output[i], EPSILON, 1.0 - EPSILON)));
                        break;
                    
                    case CCE:
                        this->error[i] = -(target[i] / max(EPSILON, activated_output[i]));
                        break;
                    
                    case HUBER:
                        double diff = target[i] - activated_output[i];
                        
                        if (abs(diff) <= delta) {
                            this->error[i] = diff; // Quadratic region
                        } else {
                            this->error[i] = delta * ((diff > 0) ? 1 : -1); // Linear region
                        }
                        break;
                }
            }
        }
        
        // Improved classification_loss_function
        void classification_loss_function() {
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
                        sum_loss += -(probability_target[i] * log(clamp(probability[i], EPSILON, 1.0 - EPSILON)) + 
                        (1 - probability_target[i]) * log(1 - clamp(probability[i], EPSILON, 1.0 - EPSILON)));
                        break;
                    
                    case CCE:
                        sum_loss += -(probability_target[i] * log(max(EPSILON, probability[i])));
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
            this->loss += (sum_loss / neurons.size()); // Average loss
        }
        
        // Improved classification_loss_derivative
        void classification_loss_derivative() {
            assert(ntype == OUTPUT); // Ensure layer is output layer
            double delta = 1.0; // Huber loss delta
            
            #pragma omp parallel for
            for (size_t i = 0; i < neurons.size(); ++i) {
                switch (lossFunc) {
                    case MSE:
                        this->error[i] = (probability_target[i] - probability[i]); // Mean squared error
                        break;

                    case MAE:
                        this->error[i] = ((probability_target[i] - probability[i]) > 0 ? 1.0 : -1.0); // Mean absolute error
                        break;

                    case BCE:
                        this->error[i] = -(probability_target[i] / clamp(probability[i], EPSILON, 1.0 - EPSILON)) + 
                        ((1.0 - probability_target[i]) / (1.0 - clamp(probability[i], EPSILON, 1.0 - EPSILON)));
                        break;

                    case CCE:
                        this->error[i] = -(probability_target[i] / max(EPSILON, probability[i]));
                        break;
                    
                    case HUBER:
                        double diff = probability_target[i] - probability[i];
                        if (abs(diff) <= delta) {
                            this->error[i] = diff; // Quadratic region
                        } else {
                            this->error[i] = delta * ((diff > 0) ? 1 : -1); // Linear region
                        }
                        break;
                }
            }
        }

        void classification_gradient() {
            assert(ntype == OUTPUT);
            
            for (size_t i = 0; i < neurons.size(); i++) {
                double a_i = probability[i];
                double y_i = probability_target[i]; // assumed target is provided
                double delta = 0.0;
                
                switch (lossFunc) {
                    case MSE:{
                        // δ = (a - y) * f'(z)
                        //delta = (a_i - y_i) * neurons[i]->get_derivative_activated_output();
                        delta = error[i] * neurons[i]->get_derivative_activated_output();
                        break;
                    }
                        
                    case MAE:{
                        // δ = sign(a - y) * f'(z)
                        delta = ((a_i > y_i) ? 1.0 : (a_i < y_i) ? -1.0 : 0.0) * neurons[i]->get_derivative_activated_output();
                        //delta = ((error[i]) ? 1.0 : (probability[i] <= probability_target[i]) ? -1.0 : 0.0) * neurons[i]->get_derivative_activated_output();
                        break;
                    }
                        
                    case BCE:{
                        // BCE derivative with sigmoid activation:
                        // δ = a - y (assuming activation is sigmoid and BCE is used) 
                        delta = error[i];//delta = a_i - y_i;
                        break;
                    }
                        
                        
                    case CCE:{
                        // Cross-entropy with softmax — full Jacobian required
                        delta = 0.0;
                        
                        for (size_t j = 0; j < neurons.size(); ++j) {
                            double a_j = probability[j];//neurons[j]->get_activated_output();
                            double y_j = probability_target[j];
                            
                            double jacobian_ij = (i == j)
                            ? a_i * (1.0 - a_i)        // ∂a_i/∂z_i
                            : -a_i * a_j;              // ∂a_j/∂z_i
                            delta += (a_j - y_j) * jacobian_ij;
                        }
                        break;
                    }
                        
                    case HUBER: {
                        // Smooth transition near zero error
                        double d = error[i];
                        double delta_h = 1.0;
                        double delta_thresh = 0.51; // optional: can be param

                        if (std::abs(d) <= delta_thresh) delta_h = d;
                        else delta_h = delta_thresh * ((d > 0) ? 1.0 : -1.0);
                        
                        delta = delta_h * neurons[i]->get_derivative_activated_output();
                        break;
                    }
                }
                
                this->gradient[i] = delta;
                neurons[i]->set_gradient(this->gradient[i]);
            }
        }

        void regression_gradient() {
            assert(ntype == OUTPUT);
            
            for (size_t i = 0; i < neurons.size(); i++) {
                double a_i = neurons[i]->get_activated_output();
                double y_i = target[i]; // assumed target is provided
                double delta = 0.0;
                
                switch (lossFunc) {
                    case MSE:{
                        // δ = (a - y) * f'(z)
                        //delta = (a_i - y_i) * neurons[i]->get_derivative_activated_output();
                        delta = error[i] * neurons[i]->get_derivative_activated_output();
                        break;
                    }
                        
                    case MAE:{
                        // δ = sign(a - y) * f'(z)
                        delta = ((a_i > y_i) ? 1.0 : (a_i < y_i) ? -1.0 : 0.0) * neurons[i]->get_derivative_activated_output();
                        //delta = ((error[i]) ? 1.0 : (a_i < y_i) ? -1.0 : 0.0) * neurons[i]->get_derivative_activated_output();
                        break;
                    }
                        
                    case BCE:{
                        // BCE derivative with sigmoid activation:
                        // δ = a - y (assuming activation is sigmoid and BCE is used)
                        delta = error[i]; //delta = a_i - y_i;
                        break;
                    }
                        
                        
                    case CCE:{
                        // Cross-entropy with softmax — full Jacobian required
                        delta = 0.0;
                        
                        for (size_t j = 0; j < neurons.size(); ++j) {
                            double a_j = neurons[j]->get_activated_output();
                            double y_j = target[j];
                            
                            double jacobian_ij = (i == j)
                            ? a_i * (1.0 - a_i)        // ∂a_i/∂z_i
                            : -a_i * a_j;              // ∂a_j/∂z_i
                            delta += (a_j - y_j) * jacobian_ij;
                        }
                        break;
                    }
                        
                    case HUBER: {
                        // Smooth transition near zero error
                        double d = a_i - y_i;
                        double delta_h = 1.0;
                        double delta_thresh = 1.0; // optional: can be param

                        if (std::abs(d) <= delta_thresh) delta_h = d;
                        else delta_h = delta_thresh * ((d > 0) ? 1.0 : -1.0);
                        
                        delta = delta_h * neurons[i]->get_derivative_activated_output();
                        break;
                    }
                }
                
                this->gradient[i] = delta;
                neurons[i]->set_gradient(this->gradient[i]);
            }
        }

        void loss_function() {
            switch(task) {
                case REGRESSION:
                    regression_loss_function();
                    break;
                case CLASSIFICATION:
                    classification_loss_function();
                    break;
                default:
                    throw runtime_error("Invalid task type for loss function.");
            }
        }

        void loss_derivative() {
            switch(task) {
                case REGRESSION:
                    regression_loss_derivative();
                    break;
                case CLASSIFICATION:
                    classification_loss_derivative();
                    break;
                default:
                    throw runtime_error("Invalid task type for loss derivative.");
            }
        }

        void gradients(){
            switch(task) {
                case REGRESSION:
                    regression_gradient();
                    break;
                case CLASSIFICATION:
                    classification_gradient();
                    break;
                default:
                    throw runtime_error("Invalid task type for gradient calculation.");
            }
        }

        void gradient_layer(vector<double> gradients, vector<vector<double>>weights){
            for(size_t i = 0; i < neurons.size(); i++){
                //double gradient_sum = 0.0;

                for(size_t j = 0; j < gradients.size(); j++){
                    for(size_t k = 0; k < weights[j].size(); k++){
                        this -> gradient[i] += gradients[j] * weights[j][k];
                    }
                }

                //this -> gradient[i] = (gradient_sum * neurons[i] -> get_derivative_activated_output());
                neurons[i] -> set_gradient(this -> gradient[i]);
            }
        }

        void regularization() {
            #pragma omp parallel for
            for(size_t i = 0; i < neurons.size(); i++){
                neurons[i] -> regularizated();
            }
        }
        

        void backpropagation(){
            #pragma omp parallel for
            for(size_t i = 0; i < neurons.size(); i++){
                neurons[i] -> set_error(this -> error[i]);
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
        vector<double> get_input() { return input; }
        vector<double> get_target() { return target; }
        vector<double> get_output() { return output; }
        vector<double> get_activated_output()  { return activated_output; }
        vector<vector<double>> get_weight()  { return weight; }
        vector<double> get_bias()  { return bias; }
        double get_loss()  { return loss; }
        vector<double> get_error()  { return error; }
        vector<double> get_probability_target()  { return probability_target; }
        vector<double> get_gradient(){return gradient;}
        vector<double> get_probability()  { return probability; }
        const vector<unique_ptr<Neuron>>& get_neuron() const noexcept {return neurons;}
        NEURONTYPE get_neuron_type() const noexcept {return ntype;}

        void set_bias(vector<double>& biases){
            assert(biases.size() == neurons.size());
            for(size_t i = 0; i < neurons.size(); i++){
                neurons[i]->set_bias(biases[i]);
            }
        }
        
        void set_weight(vector<vector<double>>& weights){
            assert(weights.size() == neurons.size());
            for(size_t i = 0; i < neurons.size(); i++){
                assert(weights[i].size() == input.size());
                neurons[i]->set_weight(weights[i]);
            }
        }

        void set_gradient(vector<double>& gradient){
            for(size_t i = 0; i < neurons.size(); i++){
                neurons[i]->set_gradient(gradient[i]);
            } 
        }

        void set_passing_gradient(vector<double>& gradients){
            for(size_t i = 0; i < neurons.size(); i++){
                for(size_t j = 0; j < gradients.size(); j++){
                    gradient[i] += gradients[j];
                }
                neurons[i]->set_gradient(gradient[i]);
            } 
        }
        
        void set_error(vector<double>& errors){
            assert(errors.size() == neurons.size());
            for(size_t i = 0; i < neurons.size(); i++){
                neurons[i]->set_error(errors[i]);
            }
        }

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

        void set_target(vector<double>& targets){
            assert(targets.size() == neurons.size()); // Ensure sizes match
            this->target = targets;
            
            for (size_t i = 0; i < neurons.size(); ++i) {
                neurons[i]->set_target(targets[i]);
            }
        }

        void gaussian_labelling(vector<double> targets, double choosen_num) {
            double min = *min_element(targets.begin(), targets.end());
            double max = *max_element(targets.begin(), targets.end());

            double range = max - min;
            assert(range > 0); // Ensure valid range

            double mean = accumulate(targets.begin(), targets.end(), 0.0) / targets.size();
            double stddev = sqrt(accumulate(targets.begin(), targets.end(), 0.0, 
                [mean](double sum, double val) { return sum + (val - mean) * (val - mean); }) / targets.size());
            assert(stddev > 0); // Ensure valid standard deviation

            for(size_t i = 0; i < target.size(); i++){
                assert(i < target.size());
                probability_target[i] = exp(pow(targets[i] - choosen_num, 2) / (2 * stddev))/(sqrt(2 * M_PI * stddev * stddev));
            }
        }

        void set_softlabeling(vector<double>& labels, double resolution) {
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

        void set_hardlabeling(vector<double>& labels){
            assert(probability_target.size() == labels.size()); // Ensure sizes match
            for(size_t i = 0; i < probability_target.size(); i++){
                this -> probability_target[i] = round(labels[i]);
            }
        }

        void set_labelling(LABELLING labelling_type, const vector<double>& targets, double label_value, double resolution) {
            size_t num_classes = target.size();
            probability_target.resize(num_classes);
            
            switch (labelling_type) {
                case HARD: // Hard one-hot
                for (size_t i = 0; i < num_classes; ++i) {
                    probability_target[i] = (target[i] == label_value) ? 1.0 : 0.0;
                }
                break;

                // Soft label smoothing
                case SOFT: {
                double smooth_val = resolution / num_classes;
                for (size_t i = 0; i < num_classes; ++i) {
                    if (target[i] == label_value) {
                        probability_target[i] = 1.0 - resolution + smooth_val;
                    } else {
                        probability_target[i] = smooth_val;
                    }
                }
            }
            break;
        case GAUSSIAN: // Gaussian labeling
            {
                double mean = accumulate(target.begin(), target.end(), 0.0) / num_classes;
                double stddev = sqrt(accumulate(targets.begin(), targets.end(), 0.0, 
                [mean](double sum, double val) { return sum + (val - mean) * (val - mean); }) / targets.size());
                for (size_t i = 0; i < num_classes; ++i) {
                    probability_target[i] = exp(-pow(target[i] - label_value, 2) / (2 * stddev * stddev));
                }
            }
            break;
        default:
            throw runtime_error("Unknown labelling type.");
    }
}

        void set_step_size(int step_size){
            for(size_t i = 0; i < neurons.size(); i++){
                neurons[i] -> set_step_size(step_size);
            }
        }

        void set_task(TASK task_type) {
            this -> task = task_type;
        }
};

#endif