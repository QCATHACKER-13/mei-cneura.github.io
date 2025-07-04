//Project: NEURA 
//NEURA is the artificial neuron and network development research for the artificial intellegence for data analysis
//Searching a right structure of the neural network and optimize an artificial fast neuron in most optimal codes
//After the further development and edges, the Project Mei as neuron investigation project and Project Raiden as network
//After Project NEURA: Artificial Neuron Network is execute then escalate into a hardware mode of neuron architecture
//Developer: Christopher Emmanuelle J. Visperas, Applied Physicist Graduated

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
using namespace std;

enum class ActivationFunction{
    SIGMOID,
    RELU,
    TANH
};

class Activation{
    private:
        ActivationFunction activateFunction;

        /// Activation Functions
        double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
        double relu(double x) { return x > 0 ? x : 0.01 * x; }
        double tanh_derivative(double x) { return (1.0 - tanh(x)) * tanh(x); }
        double sigmoid_derivative(double x) { return sigmoid(x) * (1 - sigmoid(x)); }
        double relu_derivative(double x) { return x > 0 ? 1 : 0.01; }

    public:
        Activation(ActivationFunction actFunc){
            activateFunction = actFunc;
        }
        
        double activation_value(double x) {
            switch (activateFunction) {
                case ActivationFunction::SIGMOID: return sigmoid(x);
                case ActivationFunction::RELU: return relu(x);
                case ActivationFunction::TANH: return tanh(x);
                default: return sigmoid(x);
            }
        }
        
        double activation_derivative(double x) {
            switch (activateFunction) {
            case ActivationFunction::SIGMOID: return sigmoid_derivative(x);
            case ActivationFunction::RELU: return relu_derivative(x);
            case ActivationFunction::TANH: return tanh_derivative(x);
            default: return sigmoid_derivative(x);
            }
        }

        vector<double> softmax(const vector<double>& input) {
            double maxVal = *max_element(input.begin(), input.end()); // Prevents overflow
            vector<double> exp_values(input.size());
        
            // Compute exponentials
            for (size_t i = 0; i < input.size(); ++i) {
                exp_values[i] = exp(input[i] - maxVal);
            }
        
            double sum_exp = accumulate(exp_values.begin(), exp_values.end(), 0.0);
            if (sum_exp == 0) {
                cerr << "Error: sum_exp is zero. Division by zero prevented." << endl;
                return vector<double>(input.size(), 0.0);
            }
        
            vector<double> probabilities(input.size());
        
            // Compute softmax probabilities
            for (size_t i = 0; i < input.size(); ++i) {
                probabilities[i] = exp_values[i] / sum_exp;
            }
        
            return probabilities;
        }
        
};

#endif