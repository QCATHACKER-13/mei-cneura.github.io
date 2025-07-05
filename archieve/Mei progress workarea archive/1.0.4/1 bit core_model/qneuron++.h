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

#ifndef QNEURON_H
#define QNEURON_H

#pragma once

constexpr double ALPHA = 1e-3; // Adjust this value
constexpr double EPSILON = 1e-8;
constexpr double CLIP_GRAD = 1.0; // Gradient clipping limit

// Enum for layer types
enum NEURONTYPE { INPUT, HIDDEN, OUTPUT }; // Enum for neuron types
enum ACTFUNC { SIGMOID, RELU, TANH, LEAKY_RELU, ELU}; // Enum for activation functions
enum OPTIMIZER { SGD, ADAGRAD, RMSPROP, ADAM}; // Enum for optimization algorithms
enum RANDOMTYPE { UNIFORM, NORMAL}; // Enum for random number generation types
enum LEARNRATE { CONSTANT, STEPDECAY, EXPDECAY, ITDECAY}; // Enum for learning rate adjustment strategies
enum LOSSFUNC { MSE, MAE, BCE, CCE, HUBER}; // Enum for loss functions

#include <iostream>
//#include <cstdlib>   // for rand() and srand()
//#include <ctime>     // for seeding rand()
#include <cmath>     // for exp()
#include <iomanip>   // for extension debugging
#include <random>    // For random number generation
#include <vector>
#include <algorithm> // for max_element, accumulate
#include <numeric>   // for accumulate
#include <cassert>   // for assert
#include <memory>

using namespace std;

class QNeuron {
    private:
        //neuron's parameter
        int step_size, timestep = 0; // Training step parameters

        //neuron's parameter
        bool is_dropout = false; // Flag to indicate if dropout is applied
        int dropout_mask, step_size, timestep = 0; // Training step parameters
        double mean, stddev, alpha_bar_t; // Mean, standard deviation, and  for noise distribution

        //neuron's property
        vector<double> weight, input; // input data
        double output, activated_output, target, bias, error = 1.0;
        

        vector<double> gradient_weight, //gradient of weights
            beta, //momentum factor for Adam optimizer
            m_bias;//momentum in bias
        double gradient_bias; // Gradient of bias
        vector<vector<double>> momentum; // Momentum storage for optimizers
        
        // Regularization error for weights and bias
        vector<double> error_Lw; double error_Lb = 1.0;
        int lambda_bias; vector<int> lambda_weight; // Regularization parameters
        
        //Learning parameters 
        double perturbation_std, max_reward_clip,
            learning_rate, learn_rate, decay_rate;
            
        double keep_probability, drop_probability, loss;

        ACTFUNC actFunc; // Activation function type
        LEARNRATE lr_schedule; // Learning rate ad)justment strategy
        OPTIMIZER opt; // Optimization algorithm
        LOSSFUNC lossFunc; // Loss function type
        NEURONTYPE ntype; // Neuron type (input, hidden, output)
        RANDOMTYPE randomType; // Random number generation type

        double randomUniform(double min, double max) {
            assert(min < max); // Ensure valid range
            static thread_local random_device rd; // Non-deterministic random device (if available)
            static thread_local mt19937 generator(rd()); // Mersenne Twister RNG seeded by random device
            uniform_real_distribution<double> distribution(min, max); // Uniform distribution between min and max
            return distribution(generator);
        }

        double randomNormalDist(){
            static thread_local random_device rd; // Non-deterministic random device (if available)
            static thread_local mt19937 generator(rd()); // Mersenne Twister RNG seeded by random device
            normal_distribution<double> distribution(mean, stddev); // Uniform distribution between min and max
            return distribution(generator);
        }

        int randomBernouli() {
            assert(keep_probability >= 0.0 && keep_probability <= 1.0); // Ensure valid probability range
            static thread_local random_device rd; // Non-deterministic random device (if available)
            static thread_local mt19937 generator(rd()); // Mersenne Twister RNG seeded by random device
            bernoulli_distribution distribution(keep_probability); // Bernoulli distribution with probability `p`
            return distribution(generator); // Returns 1 with probability `p`, otherwise 0
        }

        void dropout_switch(){
            dropout_mask = randomBernouli();
        }

        double dropout_feedforward(double x){
            return (is_dropout) ? x : ((x * dropout_mask)/keep_probability);
            //return (x * dropout_mask)/keep_probability; // Apply dropout during feedforward
        }

        double dropout_backpropagation(double x){
            return (is_dropout) ? x : (x * dropout_mask);
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
                    break;
                    
                case STEPDECAY:
                    if (step_size > 0 && timestep % step_size == 0) {
                        learning_rate *= decay_rate; // Prevent underflow
                    }
                    break;
                
                case EXPDECAY:
                    learning_rate *= exp(-decay_rate * timestep);
                    break;
                    
                case ITDECAY:
                    learning_rate /= (1 + (decay_rate * timestep));
                    break;
            }
        }

        void update_weights() {
            #pragma omp parallel for
            for (size_t i = 0; i < weight.size(); ++i) {
                assert(i < weight.size()); // Ensure index is within bounds
                gradient_weight[i] =  max(-CLIP_GRAD, 
                min(-(error * activation_derivative(output) * 
                        dropout_backpropagation(activation_value(output)) * input[i]) + error_Lw[i], 
                    CLIP_GRAD)); // Gradient clipping
        
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
            gradient_bias = max(-CLIP_GRAD, 
                min(-(error* activation_derivative(output)+
                dropout_backpropagation(activation_value(output))) + error_Lb, 
                    CLIP_GRAD));// Gradient clipping

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

        
    public:
        QNeuron(vector<double> inputs, 
            const double& learn_rate, const double& decay_rate, 
            const vector<double>& beta,
            double perturbation_std, double max_clip, 
            NEURONTYPE ntype, RANDOMTYPE randomType, ACTFUNC actFunc, 
            LEARNRATE lr_schedule, OPTIMIZER opt, LOSSFUNC lossFunc)
        :  input(inputs),
        perturbation_std(perturbation_std), max_reward_clip(max_clip),
        learning_rate(learn_rate), decay_rate(decay_rate), beta(beta),
        ntype(ntype), randomType(randomType), actFunc(actFunc), lr_schedule(lr_schedule), opt(opt), lossFunc(lossFunc)
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

        void He_Initialization(size_t output_size = 0) {
            weight.assign(input.size(), 0.0);

            double scale = sqrt(2.0 / input.size());  // He Initialization

            // Initialize weights with the chosen scaling
            for (size_t i = 0; i < weight.size(); ++i) {
                assert(i < weight.size()); // Ensure index is within limits
                weight[i] = randomUniform(-scale, scale);
            }
        
            // Initialize bias separately
            bias = 0.0;  // Small positive bias to prevent dead neurons
        }

        void Xavier_Initialization(size_t output_size = 0) {
            weight.assign(input.size(), 0.0);

            double scale = sqrt(2.0 / (input.size() + output_size));  // Xavier Initialization

            // Initialize weights with the chosen scaling
            for (size_t i = 0; i < weight.size(); ++i) {
                assert(i < weight.size()); // Ensure index is within limits
                weight[i] = randomUniform(-scale, scale);
            }
        
            // Initialize bias separately
            bias = randomUniform(-scale, scale);  // Xavier-based random bias for other activations
        }

        void Gaussian_Initialization() {
            // Initialize mean and standard deviation for noise distribution
            this -> mean = accumulate(input.begin(), input.end(), 0.0) / input.size();
            this -> stddev = sqrt(accumulate(input.begin(), input.end(), 0.0, 
            [this](double sum, double val) {
                return sum + (val - mean) * (val - mean);
            }) / input.size());
            
            // Initialize weights with the chosen scaling
            for (size_t i = 0; i < weight.size(); ++i) {
                assert(i < weight.size()); // Ensure index is within limits
                weight[i] = randomNormalDist();
            }
            bias = randomNormalDist(); // Initialize bias with normal distribution
        }

        // Computes the neuron's output using feedforward propagation
        void feedforward() {
            this->output = inner_product(input.begin(), input.end(), weight.begin(), bias);

            //assert(!isnan(output)); // Check for NaN output
            this->activated_output = activation_value(output);
        }
        
        // Getter functions for neuron parameters
        int get_step_size() {return step_size;}
        vector <double> get_input() {return input;}
        double get_output() {return output;}
        double get_activated_output() {return activated_output;}
        double get_bias() {return bias;}
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
};

#endif