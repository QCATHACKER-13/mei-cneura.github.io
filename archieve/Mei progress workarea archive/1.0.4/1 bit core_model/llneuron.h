#ifndef NEURON_NOPROP_H
#define NEURON_NOPROP_H

#include <vector>
#include <random>
#include <cmath>
#include <cassert>

using namespace std;

constexpr double EPSILON = 1e-8;

class NoPropNeuron {
private:
    size_t input_dim;
    vector<double> weight;
    vector<double> input;
    vector<double> target_embedding;
    vector<double> z_t_minus_1;
    vector<double> noise;
    double eta = 0.1;

    // Optimizer parameters (simple SGD)
    double learning_rate = 1e-3;

    // Helper RNGs
    default_random_engine gen;
    normal_distribution<double> normal_dist{0.0, 1.0};

    vector<double> generate_noise(size_t dim) {
        vector<double> z(dim);
        for (auto& val : z)
            val = normal_dist(gen);
        return z;
    }

    vector<double> add_noise(const vector<double>& embedding, double alpha_bar_t) {
        assert(embedding.size() == input_dim);
        vector<double> z(input_dim);
        for (size_t i = 0; i < input_dim; ++i) {
            z[i] = sqrt(alpha_bar_t) * embedding[i] + sqrt(1 - alpha_bar_t) * normal_dist(gen);
        }
        return z;
    }

    // Dummy residual block approximation û_θ(z_{t-1}, x)
    vector<double> residual_block(const vector<double>& z_prev, const vector<double>& x) {
        vector<double> result(input_dim);
        for (size_t i = 0; i < input_dim; ++i) {
            result[i] = tanh(z_prev[i] + x[i] + weight[i]);
        }
        return result;
    }

    void sgd_update(const vector<double>& predicted, const vector<double>& target) {
        for (size_t i = 0; i < input_dim; ++i) {
            double grad = predicted[i] - target[i];
            weight[i] -= learning_rate * grad;
        }
    }

public:
    NoPropNeuron(size_t dim, double lr) : input_dim(dim), learning_rate(lr) {
        weight.resize(dim, 0.01); // Simple init
    }

    void set_input(const vector<double>& x) {
        assert(x.size() == input_dim);
        input = x;
    }

    void set_target_embedding(const vector<double>& embedding) {
        assert(embedding.size() == input_dim);
        target_embedding = embedding;
    }

    void train_step(double alpha_bar_t) {
        z_t_minus_1 = add_noise(target_embedding, alpha_bar_t);
        vector<double> predicted = residual_block(z_t_minus_1, input);
        sgd_update(predicted, target_embedding);
    }

    vector<double> denoise(const vector<double>& noisy_label) {
        return residual_block(noisy_label, input);
    }

    vector<double> get_weights() const { return weight; }
};

#endif // NEURON_NOPROP_H
