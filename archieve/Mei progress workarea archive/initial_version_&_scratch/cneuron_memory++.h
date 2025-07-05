/* Project: NEURA 
NEURA is an artificial neuron and network development research project 
focused on artificial intelligence for data analysis. The goal is to find 
the optimal neural network structure and improve artificial neurons for 
efficiency and speed.

Future projects:
- Project Mei: Neuron investigation project
- Project Raiden: Network development 
- Hardware integration: After successfully developing the artificial 
  neuron network, the project will transition into hardware implementation.

Developer: Christopher Emmanuelle J. Visperas, Applied Physics Researcher*/

#ifndef CNEURON_MEMORY_H
#define CNEURON_MEMORY_H

#include <vector>
#include <cmath>
#include <random>
#include <iostream>

class MemoryNeuron {
public:
    std::vector<double> weights;
    double bias;
    double memory_cell;  // For LSTM/GRU
    double prev_output;
    double learning_rate;

    // LSTM Gates
    double forget_gate, input_gate, output_gate, cell_input;

    MemoryNeuron(size_t input_size, double lr = 0.01)
        : weights(input_size), bias(0.0), memory_cell(0.0), prev_output(0.0),
          forget_gate(0.0), input_gate(0.0), output_gate(0.0), cell_input(0.0),
          learning_rate(lr) {
        initialize();
    }

    void initialize() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        for (auto& w : weights) {
            w = dist(gen);
        }
        bias = dist(gen);
    }

    // Activation function (Sigmoid)
    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    // LSTM Forward Pass
    double LSTM_forward(const std::vector<double>& inputs) {
        double weighted_sum = 0.0;
        for (size_t i = 0; i < weights.size(); ++i) {
            weighted_sum += weights[i] * inputs[i];
        }

        forget_gate = sigmoid(weighted_sum + bias);
        input_gate = sigmoid(weighted_sum + bias);
        output_gate = sigmoid(weighted_sum + bias);
        cell_input = tanh(weighted_sum + bias);

        memory_cell = forget_gate * memory_cell + input_gate * cell_input;
        prev_output = output_gate * tanh(memory_cell);
        
        return prev_output;
    }

    // GRU Forward Pass
    double GRU_forward(const std::vector<double>& inputs) {
        double weighted_sum = 0.0;
        for (size_t i = 0; i < weights.size(); ++i) {
            weighted_sum += weights[i] * inputs[i];
        }

        double reset_gate = sigmoid(weighted_sum + bias);
        double update_gate = sigmoid(weighted_sum + bias);
        double candidate_hidden = tanh(weighted_sum * reset_gate + bias);

        prev_output = (1 - update_gate) * candidate_hidden + update_gate * prev_output;
        return prev_output;
    }

    // Hebbian Learning Rule
    void hebbian_update(const std::vector<double>& inputs, double target) {
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] += learning_rate * inputs[i] * target;
        }
    }

    // Reinforcement Learning (Q-Learning Simplified)
    void reinforce_update(const std::vector<double>& inputs, double reward) {
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] += learning_rate * reward * inputs[i];
        }
    }

    // Echo State Network (Reservoir Computing)
    double ESN_forward(const std::vector<double>& inputs) {
        double weighted_sum = 0.0;
        for (size_t i = 0; i < weights.size(); ++i) {
            weighted_sum += weights[i] * inputs[i];
        }

        prev_output = tanh(weighted_sum + bias);
        return prev_output;
    }
};

#endif // CNEURON_MEMORY_H
