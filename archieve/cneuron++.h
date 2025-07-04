/* Project: NEURA 
NEURA is an artificial neuron and network development research project 
focused on artificial intelligence for data analysis. The goal is to find 
the optimal neural network structure and improve artificial neurons for 
efficiency and speed.

Project Development and Innovations:
- Project Neura: Neuron investigation project
- Project Mei: Neural Net Architecture investigation project

Developer: Christopher Emmanuelle J. Visperas, Applied Physics Researcher*/


#ifndef CNEURONPLUS_H
#define CNEURONPLUS_H
#define ALPHA 0.01 // Adjust this value


#include <cstdlib>   // for rand() and srand()
#include <ctime>     // for seeding rand()
#include <cmath>     // for exp()
#include <iostream>  // for debugging
#include <iomanip>   // for extension debugging
#include <random>    // For random number generation
#include <vector>
#include <algorithm> // for max_element, accumulate
#include <numeric>   // for accumulate
#include <cassert>   // for assert
//#include "cmatrix.h"

using namespace std;

// Enum for activation functions
enum actfunc { SIGMOID, RELU, TANH, LEAKY_RELU, ELU};

// Enum for optimization algorithms
enum optimizer { SGD, ADAGRAD, RMSPROP, ADAM };

// Enum for learning rate adjustment strategies
enum lrs { CONSTANT, STEPDECAY, EXPDECAY, ITDECAY};

// The Neuron class represents a single artificial neuron with support 
// for various activation functions, optimizers, and learning rate schedules.
// It includes features such as batch normalization, momentum-based optimization,
// and adaptive learning rate updates.

class Neuron {
    private:
        int step_size, timestep = 0; // Training step parameters
        vector<double> weight, input, beta, m_bias, learning_rate; // Weights, inputs, and momentum-related variables
        vector<vector<double>> momentum; // Momentum storage for optimizers
        double output, target, bias, error, prediction, decay_rate;
        const double epsilon = 1e-8; // Small value to prevent division by zero

        bool use_batch_norm; // Flag for batch normalization
        double running_mean = 0.0, running_var = 1.0; // Batch normalization parameters
        double bn_gamma = 1.0, bn_beta = 0.0; // Trainable parameters for batch normalization
        
        actfunc actFunc; // Activation function type
        lrs lr_schedule; // Learning rate adjustment strategy
        optimizer opt; // Optimization algorithm

        double randomInRange(double min, double max) {
            assert(min < max); // Ensure valid range
            static thread_local random_device rd; // Non-deterministic random device (if available)
            static thread_local mt19937 generator(rd()); // Mersenne Twister RNG seeded by random device
            uniform_real_distribution<double> distribution(min, max); // Uniform distribution between min and max
            return distribution(generator);
        }

        void initialize() {
            double scale = 1.0;
        
            // Choose appropriate scaling factor based on activation function
            if (actFunc == RELU || actFunc == LEAKY_RELU) {
                scale = sqrt(2.0 / input.size());  // He Initialization
            } else if (actFunc == SIGMOID || actFunc == TANH) {
                scale = sqrt(1.0 / input.size());  // Xavier Initialization
            }
        
            // Initialize weights with the chosen scaling
            for (size_t i = 0; i < input.size(); ++i) {
                weight[i] = randomInRange(-scale, scale);
            }
        
            // Initialize bias separately
            if (actFunc == RELU || actFunc == LEAKY_RELU) {
                bias = 0.01;  // Small positive bias to prevent dead neurons
            } else {
                bias = randomInRange(-scale, scale);  // Xavier-based random bias for other activations
            }
        }

        // Activation functions and their derivatives
        static double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
        static double relu(double x) { return max(0.0, x); }
        static double leaky_relu(double x) { return x > 0 ? x : ALPHA * x; }
        static double elu(double x) { return x >= 0 ? x : ALPHA * (exp(x) - 1); }
        
        static double sigmoid_derivative(double x) { double s = sigmoid(x); return (s - pow(s, 2)); }
        static double relu_derivative(double x) { return x > 0 ? 1.0 : ALPHA; }
        static double leaky_relu_derivative(double x) { return x > 0 ? 1.0 : ALPHA; }
        static double tanh_derivative(double x) { double t = tanh(x); return 1 - pow(t, 2); }
        static double elu_derivative(double x) { return x > 0 ? 1.0 : ALPHA * exp(x); }
        
        double activation_value(double x) {
            switch (actFunc) {
                case SIGMOID: return sigmoid(x);
                case RELU: return relu(x);
                case TANH: return tanh(x);
                case LEAKY_RELU: return leaky_relu(x);
                case ELU: return elu(x);
                default: return x;
            }
        }
        
        double activation_derivative(double x) {
            switch (actFunc) {
                case SIGMOID: return sigmoid_derivative(x);
                case RELU: return relu_derivative(x);
                case TANH: return tanh_derivative(x);
                case LEAKY_RELU: return leaky_relu_derivative(x);
                case ELU: return elu_derivative(x);
                default: return 1.0;
            }
        }

        // Updates the learning rate based on the selected scheduling method
        void update_learning_rate() {
            switch (lr_schedule) {
                case CONSTANT:
                    // Keep learning rate unchanged
                    break;
                    
                case STEPDECAY:
                    /*if (timestep % step_size == 0) {
                        learning_rate *= decay_rate;  // Reduce LR at step intervals
                    }*/
                    if (step_size > 0 && timestep % step_size == 0) { // Prevent division by zero
                        learning_rate[1] = max(learning_rate[1] * decay_rate, 0.1*learning_rate[0]); // Prevent underflow
                    }
                    break;
                
                case EXPDECAY:
                    //learning_rate = learning_rate * exp(-decay_rate * timestep);
                    //learning_rate = learning_rate * exp(-decay_rate * timestep / 100.0);
                    learning_rate[1] = max(learning_rate[1] * exp(-decay_rate * timestep / 100.0), 0.1*learning_rate[0]);
                    break;
                    
                case ITDECAY:
                    //learning_rate = learning_rate / (1 + decay_rate * timestep);
                    learning_rate[1] = max(learning_rate[1] / (1 + decay_rate * timestep), 0.1*learning_rate[0]);
                    break;
            }
        }

    public:
        // Constructor for initializing neuron properties
        Neuron(int step_size, const vector<double>& inputs, const vector<double>& biasRange, const vector<double>& weightRange, double targets, 
            const vector<double>& momentum_learningRate, const vector<double> learning_rate, double decay_rate,
            actfunc actFunc, lrs lr_schedule, optimizer opt)
            
            : step_size(step_size), input(inputs), target(targets), beta(momentum_learningRate), 
            learning_rate(learning_rate), decay_rate(decay_rate),
            actFunc(actFunc), lr_schedule(lr_schedule), opt(opt)
            
            {
            
            // Initialize weight and bias vectors
            //weight.resize(inputs.size());
            srand(time(0));
            assert(!inputs.empty()); // Ensure inputs are non-empty
            weight.resize(inputs.size());

            /*for (size_t i = 0; i < inputs.size(); ++i) {
                weight[i] = randomInRange(weightRange[0], weightRange[1]);
            }
            bias = randomInRange(biasRange[0], biasRange[1]);*/

            /*Initialize momentum and m_bias vectors
            momentum.resize(2, vector<double>(inputs.size(), 0.0));
            m_bias.resize(2, 0.0);*/
            momentum.assign(2, vector<double>(inputs.size(), 0.0));
            m_bias.assign(2, 0.0);

            // Initialize weights and bias
            initialize();
        }

        // Applies batch normalization to the inputs
        void batch_normalize() {
            double mean = accumulate(input.begin(), input.end(), 0.0) / input.size();
            double variance = 0.0;
            
            for (double x : input) {
                variance += pow(x - mean, 2);
            }
            variance /= input.size();
            
            // Update running statistics for inference
            running_mean = 0.9 * running_mean + 0.1 * mean;
            running_var = 0.9 * running_var + 0.1 * variance;
            
            // Normalize and apply scale + shift
            for (size_t i = 0; i < input.size(); ++i) {
                input[i] = bn_gamma * ((input[i] - mean) / sqrt(variance + epsilon)) + bn_beta;
            }

            // Train gamma and beta instead of using constants
            bn_gamma -= learning_rate[1] * (bn_gamma - 1.0);  // Slowly adapt gamma
            bn_beta -= learning_rate[1] * bn_beta;  // Slowly adapt beta
        }

        // Computes the neuron's output using feedforward propagation
        void feedforward() {
            assert(!weight.empty()); // Ensure weights exist
            if (use_batch_norm) { batch_normalize();}
        
            /*output = (dot_product(input, weight) + bias);
            prediction = activation_value(output);  // Apply activation function
            error = target - prediction;  // Calculate error AFTER activation*/
            output = inner_product(input.begin(), input.end(), weight.begin(), 0.0) + bias;

            prediction = activation_value(output);
            error = target - output;
        }

        /*void backward() {
            update_learning_rate();
            timestep++;
            error = target - prediction;
            
            for (size_t i = 0; i < input.size(); ++i) {
                double gradient = error * input[i]; 
                
                // Compute first and second moment estimates
                momentum[0][i] = beta[0] * momentum[0][i] + (1 - beta[0]) * gradient;
                momentum[1][i] = beta[1] * momentum[1][i] + (1 - beta[1]) * pow(gradient, 2);
                
                // Bias correction
                double beta1_correction = max(1 - pow(beta[0], timestep), epsilon);
                double beta2_correction = max(1 - pow(beta[1], timestep), epsilon);
                double m_hat = momentum[0][i] / beta1_correction;
                double v_hat = momentum[1][i] / beta2_correction;

                //double m_hat = momentum[0][i] / (1 - pow(beta[0], timestep));
                //double v_hat = momentum[1][i] / (1 - pow(beta[1], timestep));
                
                // Adam weight update
                //weight[i] -= (learning_rate * m_hat) / (sqrt(v_hat) + 1e-8);
                weight[i] -= (learning_rate * m_hat) / (sqrt(v_hat) + epsilon);

            }

            //Update bias
            //double bias_gradient = error;
            //m_bias[0] = beta[0] * m_bias[0] + (1 - beta[0]) * bias_gradient;
            //m_bias[1] = beta[1] * m_bias[1] + (1 - beta[1]) * pow(bias_gradient, 2);

            //double m_bias_hat = m_bias[0] / (1 - pow(beta[0], timestep));
            //double v_bias_hat = m_bias[1] / (1 - pow(beta[1], timestep));

            //bias -= (learning_rate * m_bias_hat) / (sqrt(v_bias_hat) + 1e-8);

            // Update bias
            double bias_gradient = error;
            m_bias[0] = beta[0] * m_bias[0] + (1 - beta[0]) * bias_gradient;
            m_bias[1] = beta[1] * m_bias[1] + (1 - beta[1]) * pow(bias_gradient, 2);
            
            double m_bias_hat = m_bias[0] / max(1 - pow(beta[0], timestep), epsilon);
            double v_bias_hat = m_bias[1] / max(1 - pow(beta[1], timestep), epsilon);
            bias -= (learning_rate * m_bias_hat) / (sqrt(v_bias_hat) + epsilon);
            
        }*/
        
        // Performs backpropagation to update weights and bias
        void backward() {
            assert(timestep >= 0); // Ensure valid timestep
            timestep++;
            update_learning_rate();  // Adjust learning rate dynamically
            
            for (size_t i = 0; i < input.size(); ++i) {
                //double gradient = error * input[i];
                double gradient = - error * activation_derivative(output) * input[i]; // Compute gradient
                //cout << "Before Update: W[" << i << "] = " << weight[i] << ", Gradient = " << gradient << endl;
                
                switch (opt) {
                    case SGD:
                        //weight[i] -= learning_rate[1] * gradient;
                        momentum[0][i] = beta[0] * momentum[0][i] + (1 - beta[0]) * gradient;
                        weight[i] -= learning_rate[1] * momentum[0][i];
                        break;
                    
                    case ADAGRAD:
                        momentum[0][i] += pow(gradient, 2);
                        weight[i] -= learning_rate[1] * gradient / (sqrt(momentum[0][i]) + epsilon);
                        break;
                        
                    case RMSPROP:
                        momentum[0][i] = beta[0] * momentum[0][i] + (1 - beta[0]) * pow(gradient, 2);
                        weight[i] -= learning_rate[1] * gradient / (sqrt(momentum[0][i] + epsilon));
                        break;
                        
                    case ADAM:
                        // Compute first and second moment estimates
                        momentum[0][i] = beta[0] * momentum[0][i] + (1 - beta[0]) * gradient;
                        momentum[1][i] = beta[1] * momentum[1][i] + (1 - beta[1]) * pow(gradient, 2);
                        
                        // Correct bias
                        double m_hat = momentum[0][i] / (1 - pow(beta[0], timestep));
                        double v_hat = momentum[1][i] / (1 - pow(beta[1], timestep));
                        
                        // Apply Adam update rule
                        weight[i] -= learning_rate[1] * m_hat / (sqrt(v_hat) + epsilon);
                        break;
                    }
                    //cout << "After Update: W[" << i << "] = " << weight[i] << endl;
            }

            // Bias update (same logic as weight updates)
            double bias_gradient = - error * activation_derivative(output);
            
            switch (opt) {
                case SGD:
                    bias -= learning_rate[1] * bias_gradient;
                    break;
                
                case ADAGRAD:
                    m_bias[0] += pow(bias_gradient, 2);
                    bias -= learning_rate[1] * bias_gradient / (sqrt(m_bias[0]) + epsilon);
                    break;
                    
                case RMSPROP:
                    m_bias[0] = beta[0] * m_bias[0] + (1 - beta[0]) * pow(bias_gradient, 2);
                    bias -= learning_rate[1] * bias_gradient / (sqrt(m_bias[0]) + epsilon);
                    break;
                    
                case ADAM:
                    m_bias[0] = beta[0] * m_bias[0] + (1 - beta[0]) * bias_gradient;
                    m_bias[1] = beta[1] * m_bias[1] + (1 - beta[1]) * (bias_gradient * bias_gradient);
                    
                    //double m_bias_hat = m_bias[0] / (1 - pow(beta[0], timestep));
                    //double v_bias_hat = m_bias[1] / (1 - pow(beta[1], timestep));
                    double beta1_correction = max(1 - pow(beta[0], timestep), epsilon);
                    double beta2_correction = max(1 - pow(beta[1], timestep), epsilon);
                    double m_bias_hat = m_bias[0] / beta1_correction;
                    double v_bias_hat = m_bias[1] / beta2_correction;
                    
                    //bias -= learning_rate[1] * m_bias_hat / (sqrt(v_bias_hat) + epsilon);
                    bias -= (learning_rate[1] * m_bias_hat) / (sqrt(max(v_bias_hat, epsilon)) + epsilon);
                    break;
            }
        }

        // Prints the neuron's current state (useful for debugging)
        void print_neuron(size_t id) {
            int col1_width = 10, col2_width = 15, col3_width = 20;

            // Print table row
            cout << id << setw(col1_width) << output 
            << setw(col2_width) << bias;
            for (double w : weight) {
                cout << setw(col3_width) << w;
            }
            cout << setw(col3_width) << prediction;
            cout << setw(col3_width) << error << endl;
        }

        // Trains the neuron for a given number of epochs until the error is below a threshold
        void training(size_t id, int num_epochs, double error_margin, bool switcher) {
            vector<double> error_history;  
            
            for (int epoch = 1; epoch <= num_epochs; ++epoch) {
                feedforward();
                error_history.push_back(error);  // Store error history
                
                if (switcher) {
                    print_neuron(id);
                }
                
                if (abs(error) < error_margin) {
                    //cout << "Training complete. Error margin reached." << endl;
                    break;
                }
                
                backward();
                /*Print every 100 epochs
                if (epoch % 100 == 0) {
                    cout << "Epoch " << epoch << ": Avg Error = " 
                    << accumulate(error_history.end() - min(100, (int)error_history.size()), error_history.end(), 0.0) / min(100, (int)error_history.size()) 
                    << endl;
                }*/
            }
        }

        // Getter functions for neuron parameters
        double get_output() {return output;}

        double get_bias() {return bias;}

        vector<double> get_weight() {return weight;}

        double get_error() {return error;}

        // Setter functions for neuron parameter

        //Setting the weight of the neuron
        void set_weight(const vector<double>& weights) {
            /*if (weights.size() == weight.size()) {
                weight = weights;
            }*/
           assert(weights.size() == weight.size()); // Ensure sizes match
           weight = weights;
        }

        void set_bias(double biases) {bias = biases;} //Setting the bias of the neuron
        void set_use_batch_norm(bool use) {use_batch_norm = use;}  //Setting the batch normalization flag
};

#endif