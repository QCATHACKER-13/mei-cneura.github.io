#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <cassert>

using std::vector;
using std::cout;
using std::endl;

// Utility functions
double randn(double mean = 0.0, double stddev = 1.0) {
    static std::mt19937 gen{std::random_device{}()};
    std::normal_distribution<double> dist(mean, stddev);
    return dist(gen);
}

// Dense Layer
class LayerDense {
public:
    vector<vector<double>> weights;
    vector<double> biases;
    vector<vector<double>> output;
    vector<vector<double>> inputs;
    vector<vector<double>> dweights;
    vector<double> dbiases;
    vector<vector<double>> dinputs;

    LayerDense(int n_inputs, int n_neurons) {
        weights.resize(n_inputs, vector<double>(n_neurons));
        biases.resize(n_neurons, 0.0);
        for (auto& row : weights)
            for (auto& w : row)
                w = 0.01 * randn();
    }

    void forward(const vector<vector<double>>& inputs_) {
        inputs = inputs_;
        output.resize(inputs.size(), vector<double>(biases.size(), 0.0));
        for (size_t i = 0; i < inputs.size(); ++i)
            for (size_t j = 0; j < biases.size(); ++j) {
                output[i][j] = biases[j];
                for (size_t k = 0; k < inputs[0].size(); ++k)
                    output[i][j] += inputs[i][k] * weights[k][j];
            }
    }

    void backward(const vector<vector<double>>& dvalues) {
        dweights.assign(weights.size(), vector<double>(weights[0].size(), 0.0));
        dbiases.assign(biases.size(), 0.0);
        dinputs.assign(inputs.size(), vector<double>(weights.size(), 0.0));
        // dweights
        for (size_t i = 0; i < weights.size(); ++i)
            for (size_t j = 0; j < weights[0].size(); ++j)
                for (size_t k = 0; k < inputs.size(); ++k)
                    dweights[i][j] += inputs[k][i] * dvalues[k][j];
        // dbiases
        for (size_t j = 0; j < biases.size(); ++j)
            for (size_t k = 0; k < inputs.size(); ++k)
                dbiases[j] += dvalues[k][j];
        // dinputs
        for (size_t i = 0; i < inputs.size(); ++i)
            for (size_t j = 0; j < weights.size(); ++j)
                for (size_t k = 0; k < weights[0].size(); ++k)
                    dinputs[i][j] += dvalues[i][k] * weights[j][k];
    }
};

// ReLU Activation
class ActivationReLU {
public:
    vector<vector<double>> output;
    vector<vector<double>> inputs;
    vector<vector<double>> dinputs;

    void forward(const vector<vector<double>>& inputs_) {
        inputs = inputs_;
        output = inputs;
        for (auto& row : output)
            for (auto& val : row)
                val = std::max(0.0, val);
    }

    void backward(const vector<vector<double>>& dvalues) {
        dinputs = dvalues;
        for (size_t i = 0; i < inputs.size(); ++i)
            for (size_t j = 0; j < inputs[0].size(); ++j)
                if (inputs[i][j] <= 0)
                    dinputs[i][j] = 0.0;
    }
};

// Softmax Activation
class ActivationSoftmax {
public:
    vector<vector<double>> output;
    vector<vector<double>> dinputs;

    void forward(const vector<vector<double>>& inputs) {
        output.resize(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            double max_input = *std::max_element(inputs[i].begin(), inputs[i].end());
            vector<double> exp_values(inputs[i].size());
            double sum = 0.0;
            for (size_t j = 0; j < inputs[i].size(); ++j) {
                exp_values[j] = std::exp(inputs[i][j] - max_input);
                sum += exp_values[j];
            }
            output[i].resize(inputs[i].size());
            for (size_t j = 0; j < inputs[i].size(); ++j)
                output[i][j] = exp_values[j] / sum;
        }
    }

    void backward(const vector<vector<double>>& dvalues) {
        dinputs = dvalues; // Not implemented: use with cross-entropy for efficiency
    }
};

// Categorical Cross-Entropy Loss
class LossCategoricalCrossentropy {
public:
    double forward(const vector<vector<double>>& y_pred, const vector<int>& y_true) {
        double loss = 0.0;
        for (size_t i = 0; i < y_pred.size(); ++i) {
            double correct_conf = y_pred[i][y_true[i]];
            correct_conf = std::max(1e-7, std::min(1 - 1e-7, correct_conf));
            loss += -std::log(correct_conf);
        }
        return loss / y_pred.size();
    }
};

// SGD Optimizer
class OptimizerSGD {
public:
    double learning_rate;
    OptimizerSGD(double lr = 1.0) : learning_rate(lr) {}
    void update_params(LayerDense& layer) {
        for (size_t i = 0; i < layer.weights.size(); ++i)
            for (size_t j = 0; j < layer.weights[0].size(); ++j)
                layer.weights[i][j] -= learning_rate * layer.dweights[i][j];
        for (size_t j = 0; j < layer.biases.size(); ++j)
            layer.biases[j] -= learning_rate * layer.dbiases[j];
    }
};

// Example usage
int main() {
    // Example: 3 samples, 2 features, 3 classes
    vector<vector<double>> X = {{1.0, 2.0}, {0.5, -1.5}, {3.0, 0.0}};
    vector<int> y = {0, 2, 1};

    LayerDense dense1(2, 3);
    ActivationReLU relu1;
    LayerDense dense2(3, 3);
    ActivationSoftmax softmax;
    LossCategoricalCrossentropy loss;
    OptimizerSGD optimizer(0.1);

    // Forward pass
    dense1.forward(X);
    relu1.forward(dense1.output);
    dense2.forward(relu1.output);
    softmax.forward(dense2.output);

    double loss_val = loss.forward(softmax.output, y);
    cout << "Loss: " << loss_val << endl;

    // Backward pass (not full implementation for brevity)
    // You would implement backward for softmax+crossentropy and propagate gradients

    // Update parameters
    // optimizer.update_params(dense1);
    // optimizer.update_params(dense2);

    return 0;
}