#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
using namespace std;

// Utility functions
vector<vector<double>> dot(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    size_t rows = A.size(), cols = B[0].size(), inner = B.size();
    vector<vector<double>> result(rows, vector<double>(cols, 0.0));
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            for (size_t k = 0; k < inner; ++k)
                result[i][j] += A[i][k] * B[k][j];
    return result;
}

// Base Layer class
class Layer {
public:
    virtual void forward(const vector<vector<double>>& inputs) = 0;
    virtual void backward(const vector<vector<double>>& dvalues) = 0;
    virtual ~Layer() {}
};

// Dense Layer
class Layer_Dense : public Layer {
public:
    vector<vector<double>> weights, biases, inputs, dweights, dbiases, dinputs;
    
    Layer_Dense(int n_inputs, int n_neurons) {
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> dist(0, 0.01);
        
        weights.resize(n_inputs, vector<double>(n_neurons));
        biases.resize(1, vector<double>(n_neurons, 0.0));
        
        for (auto& row : weights)
            for (auto& w : row)
                w = dist(gen);
    }
    
    void forward(const vector<vector<double>>& inputs) override {
        this->inputs = inputs;
        output = dot(inputs, weights);
        for (size_t i = 0; i < output.size(); ++i)
            for (size_t j = 0; j < output[0].size(); ++j)
                output[i][j] += biases[0][j];
    }
    
    void backward(const vector<vector<double>>& dvalues) override {
        dweights = dot(inputs, dvalues);
        dbiases.resize(1, vector<double>(dvalues[0].size(), 0.0));
        for (size_t j = 0; j < dvalues[0].size(); ++j)
            for (size_t i = 0; i < dvalues.size(); ++i)
                dbiases[0][j] += dvalues[i][j];
        dinputs = dot(dvalues, weights);
    }
    
    vector<vector<double>> output;
};

// ReLU Activation
class Activation_ReLU : public Layer {
public:
    vector<vector<double>> inputs, dinputs, output;
    
    void forward(const vector<vector<double>>& inputs) override {
        this->inputs = inputs;
        output = inputs;
        
        for (auto& row : output)
            for (auto& val : row)
                if (val < 0) val = 0;
    }
    
    void backward(const vector<vector<double>>& dvalues) override {
        dinputs = dvalues;
        for (size_t i = 0; i < inputs.size(); ++i)
            for (size_t j = 0; j < inputs[0].size(); ++j)
                if (inputs[i][j] <= 0) dinputs[i][j] = 0;
    }
};

// Softmax Activation
class Activation_Softmax : public Layer {
public:
    vector<vector<double>> output;
    
    void forward(const vector<vector<double>>& inputs) override {
        output = inputs;
        for (auto& row : output) {
            double maxVal = *max_element(row.begin(), row.end());
            double sumExp = 0.0;
            for (auto& val : row) {
                val = exp(val - maxVal);
                sumExp += val;
            }
            for (auto& val : row) val /= sumExp;
        }
    }
};

// SGD Optimizer
class Optimizer_SGD {
public:
    double learning_rate;
    Optimizer_SGD(double lr = 0.01) : learning_rate(lr) {}
    void update(Layer_Dense& layer) {
        for (size_t i = 0; i < layer.weights.size(); ++i)
            for (size_t j = 0; j < layer.weights[0].size(); ++j)
                layer.weights[i][j] -= learning_rate * layer.dweights[i][j];
        for (size_t j = 0; j < layer.biases[0].size(); ++j)
            layer.biases[0][j] -= learning_rate * layer.dbiases[0][j];
    }
};

// Model class
class Model {
public:
    vector<Layer*> layers;
    Optimizer_SGD optimizer;
    
    Model(double lr = 0.01) : optimizer(lr) {}
    
    void add(Layer* layer) {
        layers.push_back(layer);
    }
    
    void forward(const vector<vector<double>>& X) {
        vector<vector<double>> layer_output = X;
        for (auto& layer : layers) {
            layer->forward(layer_output);
            if (dynamic_cast<Layer_Dense*>(layer))
                layer_output = dynamic_cast<Layer_Dense*>(layer)->output;
            else if (dynamic_cast<Activation_ReLU*>(layer))
                layer_output = dynamic_cast<Activation_ReLU*>(layer)->output;
            else if (dynamic_cast<Activation_Softmax*>(layer))
                layer_output = dynamic_cast<Activation_Softmax*>(layer)->output;
        }
    }
    
    void train(const vector<vector<double>>& X, int epochs = 10) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            forward(X);
            for (auto& layer : layers)
                if (dynamic_cast<Layer_Dense*>(layer))
                    optimizer.update(*dynamic_cast<Layer_Dense*>(layer));
            cout << "Epoch " << epoch + 1 << " completed." << endl;
        }
    }
};


