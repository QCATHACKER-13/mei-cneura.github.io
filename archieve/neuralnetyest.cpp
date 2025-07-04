#include "cneuron++.h"
#include "layer++.h"
#include <iostream>
#include <vector>

using namespace std;

int main() {
    // Define input, target, learning rate, and activation function
    vector<double> inputs = {0.5, -0.3, 0.8};
    vector<double> biasRange = {-1.0, 1.0};
    vector<double> weightRange = {-1.0, 1.0};
    vector<double> targets = {0.6}; // Single target for the output layer
    double learningRate = 0.01;
    ActivationFunction actFunc = ActivationFunction::SIGMOID; // Assuming SIGMOID is defined in activation.h

    // Create a hidden layer with 3 neurons
    Layer hiddenLayer(3, inputs, biasRange, weightRange, vector<double>(3, 0.0), learningRate, actFunc);

    // Create an output layer with 1 neuron
    Layer outputLayer(1, hiddenLayer.getOutput(), biasRange, weightRange, targets, learningRate, actFunc);

    // Perform feedforward for hidden layer
    hiddenLayer.feedforward();

    // Update the input of the output layer with the output of the hidden layer
    outputLayer = Layer(1, hiddenLayer.getOutput(), biasRange, weightRange, targets, learningRate, actFunc);

    // Perform feedforward for output layer
    outputLayer.feedforward();

    // Print neuron details of the output layer
    outputLayer.measurement();

    // Perform backpropagation for output layer
    outputLayer.backpropagation();

    // Perform backpropagation for hidden layer
    hiddenLayer.backpropagation();

    // Perform feedforward again for hidden layer
    hiddenLayer.feedforward();

    // Update the input of the output layer with the output of the hidden layer
    outputLayer = Layer(1, hiddenLayer.getOutput(), biasRange, weightRange, targets, learningRate, actFunc);

    // Perform feedforward again for output layer
    outputLayer.feedforward();

    // Print neuron details of the output layer
    outputLayer.measurement();

    return 0;
}