
#include <iostream>
#include <vector>
#include <cassert>
#include "hopfield.h"
#include "activation.h"

void test_hopfield() {
    // Sample data
    size_t num_neurons = 3;
    std::vector<double> inputs = {0.5, -0.3, 0.8};
    std::vector<double> targets = {1.0, -1.0, 1.0};
    double learning_rate = 0.1;
    ActivationFunction actFunc = ActivationFunction::SIGMOID;

    // Initialize Hopfield network
    Hopfield hopfield(num_neurons, inputs, targets, learning_rate, actFunc);

    // Call enthalpy method
    hopfield.enthalpy();

    // Check energy values (example check, actual values may vary)
    for (size_t i = 0; i < num_neurons; ++i) {
        assert(hopfield.energy[i] != 0); // Ensure energy is updated
    }

    // Call entropy method
    hopfield.entropy();

    // Check state values (example check, actual values may vary)
    for (size_t i = 0; i < num_neurons; ++i) {
        assert(hopfield.state[i] == 1 || hopfield.state[i] == -1 || hopfield.state[i] == 0); // Ensure state is normalized
    }

    std::cout << "All tests passed!" << std::endl;
}

int main() {
    test_hopfield();
    return 0;
}