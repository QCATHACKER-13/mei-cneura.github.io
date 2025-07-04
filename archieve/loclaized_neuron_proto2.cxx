#ifndef LOCAL_LEARNING_NEURON_H
#define LOCAL_LEARNING_NEURON_H

#include <vector>
#include <random>
#include <cmath>
#include <cassert>

using namespace std;

class LocalLearningNeuron {
private:
    vector<double> weights;
    vector<double> inputs;

    double bias;
    double output;

    double learning_rate;
    double perturbation_size;

    mt19937 rng;
    normal_distribution<double> perturb_dist;

    // Activation function (can be changed to RELU or ELU if needed)
    double activation(double x) {
        return tanh(x);  // Smooth non-linearity
    }

    // Compute neuron output given input and current weights
    double compute_output(const vector<double>& w, const double& b, const vector<double>& in) const {
        assert(w.size() == in.size());
        double sum = b;
        for (size_t i = 0; i < in.size(); ++i) {
            sum += w[i] * in[i];
        }
        return activation(sum);
    }

public:
    LocalLearningNeuron(size_t input_size, double lr = 0.01, double perturb = 0.01)
        : learning_rate(lr), perturbation_size(perturb),
          rng(random_device{}()), perturb_dist(0.0, perturb) {
        
        weights.resize(input_size);
        for (auto& w : weights) {
            w = ((double)rand() / RAND_MAX - 0.5) * 2.0;  // Initialize [-1, 1]
        }
        bias = 0.0;
    }

    // Perform feedforward pass
    double feedforward(const vector<double>& in) {
        inputs = in;
        output = compute_output(weights, bias, inputs);
        return output;
    }

    // Local learning update (no backprop), target is local supervision signal
    void local_update(double target) {
        // 1. Save original weights
        vector<double> w_perturbed = weights;
        double b_perturbed = bias;

        // 2. Apply small random perturbation to weights
        for (auto& w : w_perturbed) {
            w += perturb_dist(rng);
        }
        b_perturbed += perturb_dist(rng);

        // 3. Compute perturbed output
        double output_perturbed = compute_output(w_perturbed, b_perturbed, inputs);

        // 4. Compute energy (squared error) for baseline and perturbed
        double baseline_energy = 0.5 * pow(target - output, 2);
        double perturbed_energy = 0.5 * pow(target - output_perturbed, 2);

        // 5. Compare energies
        double direction = (perturbed_energy < baseline_energy) ? 1.0 : -1.0;

        // 6. Update weights in the direction of perturbation
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] += direction * learning_rate * inputs[i];
        }

        // 7. Update bias
        bias += direction * learning_rate;
    }

    // Accessors
    double get_output() const { return output; }
    vector<double> get_weights() const { return weights; }
    double get_bias() const { return bias; }
};

#endif
