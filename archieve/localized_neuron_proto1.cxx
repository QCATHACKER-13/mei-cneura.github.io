#ifndef LOCAL_LEARNING_NEURON_H
#define LOCAL_LEARNING_NEURON_H

#include <vector>
#include <random>
#include <cmath>

using namespace std;

class LocalLearningNeuron {
private:
    vector<double> weight;
    vector<double> input;
    double bias;
    double output;
    double learning_rate;
    double perturbation_size;
    mt19937 rng;
    normal_distribution<double> noise;

    double activation(double x) {
        return tanh(x); // or sigmoid, depending on design
    }

    double compute_output(const vector<double>& inputs) {
        double sum = bias;
        for (size_t i = 0; i < inputs.size(); ++i) {
            sum += inputs[i] * weight[i];
        }
        return activation(sum);
    }

public:
    LocalLearningNeuron(size_t input_size, double lr = 0.01, double perturb = 0.01)
        : learning_rate(lr), perturbation_size(perturb), rng(random_device{}()), noise(0.0, perturb) {
        weight.resize(input_size);
        for (auto& w : weight) {
            w = (double(rand()) / RAND_MAX - 0.5) * 2.0; // small random init
        }
        bias = 0.0;
    }

    double feedforward(const vector<double>& inputs) {
        input = inputs;
        output = compute_output(inputs);
        return output;
    }

    void local_update(double local_target) {
        // Perturb weights slightly
        vector<double> perturbed_weight = weight;
        for (auto& w : perturbed_weight) {
            w += noise(rng);
        }

        double perturbed_output = compute_output(input); // with perturbed weights

        // Compare perturbed output to baseline
        double baseline_error = 0.5 * pow(local_target - output, 2);
        double perturbed_error = 0.5 * pow(local_target - perturbed_output, 2);

        // If perturbed error is lower, reinforce perturbation direction
        double reward = (perturbed_error < baseline_error) ? 1.0 : -1.0;

        // Update weights
        for (size_t i = 0; i < weight.size(); ++i) {
            weight[i] += reward * learning_rate * input[i];
        }
        bias += reward * learning_rate;
    }
};

#endif
