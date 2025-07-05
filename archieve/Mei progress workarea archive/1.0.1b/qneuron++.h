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

#include <iostream>
#include <cmath>     // for exp()
#include <iomanip>   // for extension debugging
#include <random>    // For random number generation
#include <vector>
#include <algorithm> // for max_element, accumulate
#include <numeric>   // for accumulate
#include <cassert>   // for assert
#include <memory>

using namespace std;

enum NEURONTYPE { DEFAULT, HIDDEN, OUTPUT }; // Enum for neuron types
enum ACTFUNC { SIGMOID, RELU, TANH, LEAKY_RELU, ELU}; // Enum for activation functions
enum OPTIMIZER { SGD, ADAGRAD, RMSPROP, ADAM}; // Enum for optimization algorithms
enum LEARNRATE { CONSTANT, STEPDECAY, EXPDECAY, ITDECAY}; // Enum for learning rate adjustment strategies
enum LOSSFUNC { MSE, BCE, CCE, HUBER}; // Enum for loss functions


class LocalLearningNeuron {
private:
    vector<double> weights;
    vector<double> inputs;

    double bias;
    double output;

    double learning_rate;
    double perturbation_std;
    double max_reward_clip;

    mt19937 rng;
    normal_distribution<double> noise_dist;

    double activation(double x) const {
        return tanh(x);  // Replace with other activation if needed
    }

    double compute_output(const vector<double>& w, const double& b, const vector<double>& in) const {
        assert(w.size() == in.size());
        double sum = b;
        for (size_t i = 0; i < in.size(); ++i) {
            sum += w[i] * in[i];
        }
        return activation(sum);
    }

public:
    LocalLearningNeuron(size_t input_size,
                        double lr = 0.01,
                        double perturb_std = 0.01,
                        double max_clip = 1.0)
        : learning_rate(lr),
          perturbation_std(perturb_std),
          max_reward_clip(max_clip),
          rng(random_device{}()),
          noise_dist(0.0, perturb_std)
    {
        weights.resize(input_size);
        for (auto& w : weights) {
            w = ((double)rand() / RAND_MAX - 0.5) * 2.0;
        }
        bias = 0.0;
    }

    double feedforward(const vector<double>& in) {
        inputs = in;
        output = compute_output(weights, bias, inputs);
        return output;
    }

    void local_update(double target) {
        double baseline_output = output;
        //MSE formula
        double baseline_energy = 0.5 * pow(target - baseline_output, 2); 

        vector<double> w_perturbed = weights;
        double b_perturbed = bias;

        for (auto& w : w_perturbed) {
            w += noise_dist(rng);
        }
        b_perturbed += noise_dist(rng);

        double perturbed_output = compute_output(w_perturbed, b_perturbed, inputs);
        double perturbed_energy = 0.5 * pow(target - perturbed_output, 2);

        double reward = baseline_energy - perturbed_energy;
        reward = max(-max_reward_clip, min(max_reward_clip, reward));

        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] += learning_rate * reward * inputs[i];
        }
        bias += learning_rate * reward;
    }

    void reinitialize_weights() {
        for (auto& w : weights) {
            w = ((double)rand() / RAND_MAX - 0.5) * 2.0;
        }
        bias = 0.0;
    }

    double get_output() const { return output; }
    const vector<double>& get_weights() const { return weights; }
    double get_bias() const { return bias; }
};

#endif