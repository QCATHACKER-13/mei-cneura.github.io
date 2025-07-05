/* Project: NEURA 
NEURA is an artificial neuron and network development research project 
focused on artificial intelligence for data analysis. The goal is to find 
the optimal neural network structure and improve artificial neurons for 
efficiency and speed.

Project Development and Innovations:
- Project Neura: Neuron investigation project
- Project Mei: Neural Net Architecture investigation project

Developer: Christopher Emmanuelle J. Visperas, Applied Physics Researcher

Note in this neuron testing in this following 
    - beta or momentum factor is set to 0.9 and 0.9999 as default, and must be the constant parameter
    - if beta is not a constant parameter then the target data with in the 
      range of the input data cause sometimes are error and more moderate error if the
      target data is out of range.


*/

#ifndef CNEURON_H
#define CNEURON_H


#pragma once

#include <iostream>
#include <cstdlib>   // for rand() and srand()
#include <ctime>     // for seeding rand()
#include <cmath>     // for exp()
#include <iomanip>   // for extension debugging
#include <random>    // For random number generation
#include <vector>
#include <algorithm> // for max_element, accumulate
#include <numeric>   // for accumulate
#include <cassert>   // for assert
#include <memory>


#include "../model_options/model_option.h"

using namespace std;

// The Neuron class represents a single artificial neuron with support 
// for various activation functions, optimizers, and learning rate schedules.
// It includes features such as batch normalization, momentum-based optimization,
// and adaptive learning rate updates.

class Neuron {
    private:
        //neuron's parameter
        bool is_dropout = false; // Flag to indicate if dropout is applied
        int dropout_mask, step_size, timestep = 0; // Training step parameters

        int lambda_bias; vector<int> lambda_weight; // Regularization parameters

        //neuron's property
        vector<double> weight, // weights on each input
            input, // input data
            gradient_weight, //gradient of weights
            beta, //momentum factor for Adam optimizer
            m_bias,//momentum in bias
            error_Lw; // Regularization error for weights
        
        vector<vector<double>> momentum; // Momentum storage for optimizers
        double output, target, bias, 
            activated_output, derivative_activated_output,
            error = 1.0, error_Lb = 1.0, gradient,
            gradient_bias, 
            learning_rate, learn_rate, decay_rate, 
            keep_probability, drop_probability,
            loss = 0.0; 
        
        vector<double> error_history; // Error history for analysis
        ACTFUNC actFunc; // Activation function type
        LEARNRATE lr_schedule; // Learning rate ad)justment strategy
        OPTIMIZER opt; // Optimization algorithm
        LOSSFUNC lossFunc; // Loss function type
        NEURONTYPE ntype; // Neuron type (input, hidden, output)
        DISTRIBUTION distribution; // Distribution type for weight initialization
        INITIALIZATION initializer;

        // Centralized random engine for all Neuron instances
        static std::mt19937& global_rng() {
            static std::random_device rd;
            static std::mt19937 rng(rd()); 
            return rng;
        }
        
        double randomInRange(double min, double max) {
            assert(min < max);
            std::uniform_real_distribution<double> dist(min, max);
            return dist(global_rng());
        }
        
        int randomBinary() {
            assert(keep_probability >= 0.0 && keep_probability <= 1.0);
            std::bernoulli_distribution dist(keep_probability);
            return dist(global_rng());
        }

        void dropout_switch(){
            dropout_mask = randomBinary();
        }

        double dropout_feedforward(double x){
            return (is_dropout) ? x : ((x * dropout_mask)/keep_probability);
            //return (x * dropout_mask)/keep_probability; // Apply dropout during feedforward
        }

        double dropout_backpropagation(double x){
            return (is_dropout) ? x : (x * dropout_mask);
            //return (x * dropout_mask);
        }

        // Activation functions and their derivatives
        double activation_value(double x) {
            switch (actFunc) {
                case SIGMOID: return 1.0 / (1.0 + exp(-x));
                case RELU: return max(0.0, x);
                case TANH: return tanh(x);
                case LEAKY_RELU: return x > 0 ? x : ALPHA * x;
                case ELU: return x >= 0 ? x : ALPHA * (exp(x) - 1);
                default: return x;
            }
        }
        
        double activation_derivative(double x) {
            switch (actFunc) {
                case SIGMOID: {
                    double s = activation_value(x);
                    return s * (1.0 - s);
                }
                case RELU: return x > 0 ? 1.0 : 0.0;
                case TANH: {
                    double t = tanh(x);
                    return 1.0 - t * t;
                }
                case LEAKY_RELU: return x > 0 ? 1.0 : ALPHA;
                case ELU: return x >= 0 ? 1.0 : ALPHA * std::exp(x);
                default: return 1.0;
            }
        }

        // Updates the learning rate based on the selected scheduling method
        void update_learning_rate() {
            switch (lr_schedule) {
                case CONSTANT:
                    // Keep learning rate unchanged
                    //initial_error = error;
                    learning_rate = learn_rate;
                    break;
                    
                case STEPDECAY:
                    if (step_size > 0 && timestep % step_size == 0) {
                        learning_rate *= decay_rate; // Prevent underflow
                        //learning_rate = learn_rate * decay_rate;
                    }
                    break;
                
                case EXPDECAY:
                    learning_rate *= exp(-decay_rate * timestep);
                    //learning_rate = learn_rate * exp(-decay_rate * timestep);
                    //learning_rate = max(min(learning_rate, 1e2), 1e-6);
                    break;
                    
                case ITDECAY:
                    learning_rate /= (1 + (decay_rate * timestep));
                    //learning_rate = learn_rate/(1 + (decay_rate * timestep));
                    break;
            }
        }

    public:
        // Constructor for initializing neuron properties
        Neuron(vector<double> inputs, 
            const double& learn_rate, const double& decay_rate, const vector<double>& beta,
            NEURONTYPE ntype, INITIALIZATION initializer, DISTRIBUTION distribution, 
            ACTFUNC actFunc, LEARNRATE lr_schedule, OPTIMIZER opt, LOSSFUNC lossFunc)
            
            : input(inputs),
            learning_rate(learn_rate), decay_rate(decay_rate), beta(beta),
            ntype(ntype), distribution(distribution), initializer(initializer), actFunc(actFunc), lr_schedule(lr_schedule), opt(opt), lossFunc(lossFunc)
            
            {
                // Initialize weight and bias vectors
                this -> input = inputs;
                assert(!inputs.empty()); // Ensure inputs are non-empty
                
                momentum.assign(2, vector<double>(inputs.size(), 0.0));
                m_bias.assign(2, 0.0);
                error_Lw.assign(inputs.size(), 1.0); // Initialize regularization error for weights
                lambda_weight.assign(inputs.size(), 0); // Initialize regularization weights

                if(ntype == OUTPUT) is_dropout = true;
        }

        void initialize(size_t output_size = 0) {
            this -> bias = 0.0;
            weight.assign(input.size(), 0.0);
            this -> gradient = 0.0;
            gradient_weight.assign(input.size(), 0.0);
            this -> gradient_bias = 0.0;
            error_Lw.resize(weight.size(), 0.0);

            double num_const, scale = 1.0;

            if (distribution == NORMAL) num_const = 2.0; // Standard deviation for normal distribution
            else if (distribution == UNIFORM) num_const = 6.0; // Range for uniform distribution
            else throw invalid_argument("Invalid distribution type");

            //Choose appropriate scaling factor based on activation function
            if (actFunc == RELU || actFunc == LEAKY_RELU) {
                scale = sqrt(num_const / input.size());  // He Initialization
            } else if (actFunc == SIGMOID || actFunc == TANH) {
                scale = sqrt(num_const / (input.size() + output_size));  // Xavier Initialization
            }

            // Initialize weights with the chosen scaling
            for (size_t i = 0; i < weight.size(); ++i) {
                assert(i < weight.size()); // Ensure index is within limits
                weight[i] = randomInRange(-scale, scale);
            }
        
            // Initialize bias separately
            if (actFunc == RELU || actFunc == LEAKY_RELU) {
                bias = randomInRange(-1, 1);  // Small positive bias to prevent dead neurons
            } else {
                bias = randomInRange(-scale, scale);  // Xavier-based random bias for other activations
            }
        }

        void initialization(size_t output_size = 0) {
            this -> timestep = 0; // Reset timestep for new initialization
            this -> bias = 0.0;
            weight.assign(input.size(), 0.0);
            this -> gradient = 0.0;
            gradient_weight.assign(input.size(), 0.0);
            this -> gradient_bias = 0.0;
            error_Lw.resize(weight.size(), 0.0);

            double num_const, scale = 1.0;

            if (distribution == NORMAL) num_const = 2.0; // Standard deviation for normal distribution
            else if (distribution == UNIFORM) num_const = 6.0; // Range for uniform distribution
            else throw invalid_argument("Invalid distribution type");

            //Choose appropriate scaling factor based on activation function
            if (initializer == HE) {
                scale = sqrt(num_const / input.size());  // He Initialization
            } else if (initializer == XAVIER) {
                scale = sqrt(num_const / (input.size() + output_size));  // Xavier Initialization
            }

            // Initialize weights with the chosen scaling
            for (size_t i = 0; i < weight.size(); ++i) {
                assert(i < weight.size()); // Ensure index is within limits
                weight[i] = randomInRange(-scale, scale);
            }
        
            // Initialize bias separately
            if (initializer == HE) {
                bias = 0.0;  // Small positive bias to prevent dead neurons
            } else {
                bias = randomInRange(-scale, scale);  // Xavier-based random bias for other activations
            }
        }

        // Computes the neuron's output using feedforward propagation
        void feedforward() {
            this->output = inner_product(input.begin(), input.end(), weight.begin(), bias);

            //assert(!isnan(output)); // Check for NaN output
            dropout_switch(); // Apply dropout
            this->activated_output = dropout_feedforward(activation_value(output));
            this->derivative_activated_output = activation_derivative(output);
            //this->activated_output = max(min(1.0, dropout_feedforward(activation_value(output))), -1.0);
            //this->derivative_activated_output = max(min(1.0, dropout_backpropagation(activation_derivative(output))), -1.0);
        }

        void regularizated(){
            #pragma omp parallel for
            for (size_t i = 0; i < weight.size(); ++i) {
                assert(i < weight.size()); // Ensure index is within limits
                
                if(weight[i] > 0){lambda_weight[i] = 1;}
                else if(weight[i] < 0){lambda_weight[i] = -1;}
                else { lambda_weight[i] = 0; }

                this -> loss += (lambda_weight[i] * pow(weight[i], 2)) 
                    + (lambda_weight[i] * abs(weight[i]));
                this -> error_Lw[i] = (2 * lambda_weight[i] * weight[i]) + (lambda_weight[i]);
            }

            if(bias > 0){lambda_bias = 1;}
            else if(bias < 0){lambda_bias = -1;}
            else { lambda_bias = 0; }
            this -> loss += (lambda_bias * pow(bias, 2)) + (lambda_bias * abs(bias));
            this -> error_Lb = (2 * lambda_bias * bias) + (lambda_bias);
        }
        
        // Performs backpropagation to update weights and bias
        void backward() {
            assert(timestep >= 0); // Ensure valid timestep
            timestep++;
            update_learning_rate();  // Adjust learning rate dynamically
            update_weights(); // Update weights
            update_bias(); // Update bias
        }

        void update_weights() {
            #pragma omp parallel for
            for (size_t i = 0; i < weight.size(); ++i) {
                assert(i < weight.size()); // Ensure index is within bounds
                this -> gradient_weight[i] = (learning_rate * gradient * input[i] * dropout_backpropagation(activated_output))+ error_Lw[i];

                switch (opt) {
                    case SGD:
                        this->momentum[0][i] = ((beta[0] * momentum[0][i]) + ((1 - beta[0]) * gradient_weight[i]));
                        this->weight[i] -= (learning_rate * momentum[0][i]);
                        break;
        
                    case ADAGRAD:
                        this->momentum[0][i] += pow(gradient_weight[i], 2);
                        this->weight[i] -= ((learning_rate * gradient_weight[i]) / sqrt(momentum[0][i]+  EPSILON));
                        break;
        
                    case RMSPROP:
                        this->momentum[0][i] = ((beta[0] * momentum[0][i]) + ((1 - beta[0]) * pow(gradient_weight[i], 2)));
                        this->weight[i] -= ((learning_rate * gradient_weight[i]) / sqrt(momentum[0][i]+ EPSILON));
                        break;
        
                    case ADAM:
                        this->momentum[0][i] = ((beta[0] * momentum[0][i]) + ((1 - beta[0]) * gradient_weight[i]));
                        this->momentum[1][i] = ((beta[1] * momentum[1][i]) + ((1 - beta[1]) * pow(gradient_weight[i], 2)));
        
                        double m_hat = (momentum[0][i] / (1 - pow(beta[0], timestep)));
                        double v_hat = (momentum[1][i] / (1 - pow(beta[1], timestep)));
        
                        this->weight[i] -= ((learning_rate * m_hat) / sqrt(v_hat +  EPSILON));
                        break;
                }
            }
        }

        void update_bias(){
            // Bias update (same logic as weight updates)
            this -> gradient_bias = gradient + error_Lb;
            //this -> gradient_bias = (learning_rate * gradient * dropout_backpropagation(activated_output)) + error_Lb;
            //gradient_bias = -(error* activation_derivative(output) + dropout_backpropagation(activation_value(output))) + error_Lb;// Gradient clipping

            switch (opt) {
                case SGD:
                    this -> bias -= (learning_rate * gradient_bias);
                    break;
                
                case ADAGRAD:
                    this -> m_bias[0] += pow(gradient_bias, 2);
                    this -> bias -= ((learning_rate * gradient_bias) / sqrt(m_bias[0] +  EPSILON));
                    break;
                    
                case RMSPROP:
                    this -> m_bias[0] = ((beta[0] * m_bias[0]) + ((1 - beta[0]) * pow(gradient_bias, 2)));
                    this -> bias -= ((learning_rate * gradient_bias) / sqrt(m_bias[0] + EPSILON));
                    break;
                    
                case ADAM:
                    this -> m_bias[0] = ((beta[0] * m_bias[0]) + ((1 - beta[0]) * gradient_bias));
                    this -> m_bias[1] = ((beta[1] * m_bias[1]) + ((1 - beta[1]) * pow(gradient_bias, 2)));
                    
                    double beta1_correction = max(1 - pow(beta[0], timestep), EPSILON);
                    double beta2_correction = max(1 - pow(beta[1], timestep), EPSILON);
                    double m_bias_hat = (m_bias[0] / beta1_correction);
                    double v_bias_hat = (m_bias[1] / beta2_correction);
                    
                    this -> bias -= ((learning_rate * m_bias_hat) / sqrt(v_bias_hat + EPSILON));
                    break;
            }
        }

        // Getter functions for neuron parameters
        int get_step_size() {return step_size;}
        vector <double> get_input() {return input;}
        double get_output() {return output;}
        double get_activated_output() {return activated_output;}
        double get_bias() {return bias;}
        double get_derivative_activated_output(){return derivative_activated_output;}
        double get_gradient(){return gradient;}
        double get_gradient_bias() {return gradient_bias;}
        vector<double> get_gradient_weight() {return gradient_weight;}
        vector<double> get_weight() {return weight;}
        vector<double> get_error_Lw() {return error_Lw;}
        double get_error_Lb() {return error_Lb;}
        double get_error() {return error;}
        double get_learning_rate() {return learning_rate;}
        double get_timestep(){return timestep;}
        double get_loss() {return loss;}

        // Setter functions for neuron parameter
        void set_dropout(const double& keep_prob) {
            assert(keep_prob >= 0.0 || keep_prob > 1.0); // Ensure valid dropout rate
            if(keep_prob != 0.5){
                this -> keep_probability = keep_prob;
                this -> drop_probability = 1 - keep_prob;
            }
            else if(keep_prob == 0.5){
                this -> drop_probability = 0.0;
                this -> keep_probability = 1.0;
            }
        } //Setting the dropout rate
        void set_step_size(const int& stepsize){this -> step_size = stepsize;}
        void set_input(const vector<double>& inputs) {this -> input = inputs;}
        void set_target(const double& targets){this -> target = targets;} //Setting the target of the neuron
        void set_weight(const vector<double>& weights) {
            assert(weights.size() == weight.size() && weights.size() == input.size()); // Ensure sizes match
            this->weight = weights;
        }
        void set_bias(double biases) {this -> bias = biases;} //Setting the bias of the neuron
        void set_error(double errors) {this -> error = errors;} //Setting the error of the neuron
        void set_learning_rate(const double& learning_rate) {this -> learning_rate = learning_rate;} //Setting the learning rate
        void set_gradient(double gradient) {this -> gradient = gradient;}
};

#endif