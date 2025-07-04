#include "cneuron.h"
#include <iostream>
#include <vector>
#include <cassert>

using namespace std;

void initialize_neuron(Neuron &neuron) {
    // Perform feedforward
    neuron.feedforward();
    neuron.activation();
    neuron.print_neuron();

    // Perform backpropagation
    neuron.backward();

    // Perform feedforward again
    neuron.feedforward();
    neuron.activation();
    neuron.print_neuron();
}

int main() {
    // Define input, target, learning rate, and activation function
    vector<double> inputs = {0.5, -0.3, 0.8};
    double learning_rate = 0.01, target = 0.6;
    ActivationFunction actFunc = ActivationFunction::RELU; // Assuming SIGMOID is defined in activation.h

    // Create a Neuron object
    Neuron neuron(inputs, target, learning_rate, actFunc);

    // Assertions to validate assumptions
    assert(!inputs.empty());
    assert(learning_rate > 0);

    // Initialize neuron
    initialize_neuron(neuron);

    return 0;
}