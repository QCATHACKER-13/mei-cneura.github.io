/* Project: NEURA 
NEURA is an artificial neuron and network development research project 
focused on artificial intelligence for data analysis. The goal is to find 
the optimal neural network structure and improve artificial neurons for 
efficiency and speed.

Project Development and Innovations:
- Project Neura: Neuron investigation project
- Project Mei: Neural Net Architecture investigation project

Developer: Christopher Emmanuelle J. Visperas, Applied Physics Researcher*/

#ifndef CNEURA_H
#define CNEURA_H

#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <chrono>
#include <fstream>
#include <string>
#include <functional>
#include "clayer.h"
#include "cneuron.h"
#include "../data_tools/cdata_tools.h"

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::duration;

namespace neura_utils {
    inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
        assert(a.size() == b.size());
        return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
    }
    inline double norm(const std::vector<double>& v) {
        return std::sqrt(dot(v, v));
    }
    inline void add_inplace(std::vector<double>& a, const std::vector<double>& b) {
        assert(a.size() == b.size());
        for (size_t i = 0; i < a.size(); ++i) a[i] += b[i];
    }
    inline void scale_inplace(std::vector<double>& a, double s) {
        for (auto& x : a) x *= s;
    }
}

class Neural {
private:
    vector<unique_ptr<Layer>> layers;
    vector<size_t> num_neurons;
    vector<double> train_loss_history, val_loss_history;
    function<void(int, double, double)> epoch_callback; // epoch, train_loss, val_loss

public:
    Neural(const vector<size_t>& num_neurons, const vector<double>& inputs,
           const double& learning_rate, const double& decay_rate, const vector<double>& beta,
           INITIALIZATION initializer, DISTRIBUTION distribution, 
           vector<ACTFUNC> actFunc, LEARNRATE lr_schedule, OPTIMIZER opt, LOSSFUNC lossFunc) 
           : 
           num_neurons(num_neurons) {
            if (inputs.empty()) {
                throw invalid_argument("Input vector cannot be empty.");
            }
            if (num_neurons.empty()) {
                throw invalid_argument("Number of neurons per layer must not be empty.");
            }
            
            layers.reserve(num_neurons.size());
            vector<double> current_input = inputs;
            
            for (size_t i = 0; i < num_neurons.size(); ++i) {
                NEURONTYPE layer_type = (i == num_neurons.size() - 1) ? OUTPUT : HIDDEN;
                
                auto layer = make_unique<Layer>(
                    num_neurons[i], current_input,
                    learning_rate, decay_rate, beta,
                    layer_type, initializer, distribution, actFunc[i], lr_schedule, opt, lossFunc
                );
                
                if (!layer) {
                    throw runtime_error("Failed to initialize layer " + std::to_string(i));
                }
                
                layer -> initialize(); // Initialize each layer
                layer -> feedforward(); // Feedforward to set the input for the next layer
                current_input = layer -> get_activated_output(); // Set the output of the current layer as input for the next layer
                
                // Store the layer in the vector
                layers.emplace_back(std::move(layer));
            }
            if (layers.empty()) {
                throw runtime_error("No layers were initialized in the neural network.");
            }
        }

    void layer_initialization() {
        if (layers.empty()) throw runtime_error("No layers in network.");
        #pragma omp parallel for
        for (size_t i = 0; i < layers.size(); ++i) {
            layers[i]->initialize();
        }
    }

    // Perform feedforward computation
    void feedforward() {
        if (layers.empty()) throw runtime_error("No layers in network.");
        vector<double> current_input = layers.front()->get_input();
        
        #pragma omp parallel for
        for (size_t i = 0; i < layers.size(); ++i) {
            layers[i]->set_input(current_input);
            layers[i]->feedforward();
            current_input = layers[i]->get_activated_output(); // Normalize output for next layer
        }
    }

    void LayerNormalization() {
        if (layers.empty()) throw runtime_error("No layers in network.");
        #pragma omp parallel for
        for (auto& layer : layers) {
            layer->LayerNormalization();
        }
    }

    void probability() {
        if (layers.empty()) throw runtime_error("No layers in network.");
        layers.back()->softmax();
        layers.back()->loss_function();
        layers.back()->loss_derivative();
    }

    void regularization() {
        if (layers.empty()) throw runtime_error("No layers in network.");
        #pragma omp parallel for
        for (auto& layer : layers) {
            layer->regularization();
        }
    }

    void backpropagation() {
        if (layers.empty()) return;
        layers.back() -> backpropagation();
        layers.back() -> gradient_olayer();

        #pragma omp parallel for
        for (int i = static_cast<int>(layers.size()) - 2; i >= 0; --i) {
            //layers[i]->set_error(layers[i + 1]->get_error());
            layers[i] -> gradient_hlayer(
                layers[i + 1] -> get_gradient(),
                layers[i + 1] -> get_weight()
            );
            layers[i]->backpropagation();
        }
    }

    void learning(int steps, double threshold = 0.05) {
        if (layers.empty()) throw invalid_argument("The layers didn't initialize properly.");
        auto start = high_resolution_clock::now();

        layer_initialization();

        double epoch_loss = 0.0;
        
        for (int step = 1; step <= steps; ++step) {
            feedforward();
            probability();
            
            const auto& errors = layers.back()->get_error();
            bool below_threshold = all_of(errors.begin(), errors.end(), [&](double err) {
                return std::abs(err) < threshold;
            });
            
            epoch_loss += layers.back()->get_loss();
            
            if (below_threshold) {
                break;
            }
            
            backpropagation();
            regularization();
        }
        
        print();
        
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        std::cout << "Final Loss: " << layers.back()->get_loss() << std::endl;
        std::cout << "Average Loss: " << epoch_loss/steps << std::endl;
        std::cout << "Training completed in " << duration.count() << " microseconds" << std::endl;
    }

    void enforce_learning(int epochs, double threshold = 0.05) {
        if (layers.empty()) throw invalid_argument("The layers didn't initialize properly.");
        auto start = high_resolution_clock::now();
        
        double epoch_loss = 0.0;
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            
            layer_initialization();
            for (int step = 1; step <= epoch; ++step) {
                feedforward();
                probability();
                
                const auto& errors = layers.back()->get_error();
                bool below_threshold = all_of(errors.begin(), errors.end(), [&](double err) {
                    return std::abs(err) < threshold;
                });
                
                double current_loss = layers.back()->get_loss();
                epoch_loss += current_loss;
                
                if (below_threshold) {
                    print();
                    break;
                }
                
                backpropagation();
                regularization();
            }
            epoch_loss /= epoch;
        }
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        std::cout << "Final Loss: " << layers.back()->get_loss() << std::endl;
        std::cout << "Average Loss: " << epoch_loss << std::endl;
        std::cout << "Training completed in " << duration.count() << " microseconds" << std::endl;
    }


    void batch_learning(vector<vector<double>>& training_inputs,
              vector<vector<double>>& training_targets,
              int epochs,
              int batch_size,
              double threshold = 0.05) {
                
                if (layers.empty()) 
                throw runtime_error("Neural network is not initialized.");

                if (training_inputs.empty() || training_targets.empty())
                throw invalid_argument("Training data cannot be empty.");
                
                if (training_inputs.size() != training_targets.size())
                throw invalid_argument("Input and target size mismatch.");
                
                auto start = high_resolution_clock::now();
                size_t total_samples = training_inputs.size();
                
                for (int epoch = 1; epoch <= epochs; ++epoch) {
                    cout << "\nEpoch " << epoch << "/" << epochs << endl;
                    double epoch_loss = 0.0;
                    int num_batches = (total_samples + batch_size - 1) / batch_size;
                    
                    for (int batch = 0; batch < num_batches; ++batch) {
                        int start_idx = batch * batch_size;
                        int end_idx = min(start_idx + batch_size, static_cast<int>(total_samples));
                        
                        // For each sample in the batch
                        for (int i = start_idx; i < end_idx; ++i) {
                            set_input(training_inputs[i]);
                            set_target(training_targets[i]);
                            
                            feedforward();
                            probability();
                            
                            const auto& errors = get_error();
                            bool below_threshold = all_of(errors.begin(), errors.end(),
                            [&](double err) { return std::abs(err) < threshold; });
                            
                            epoch_loss += layers.back()->get_loss();
                            
                            backpropagation();
                            regularization();
                            
                            if (below_threshold) {
                                cout << "Early stopping at sample " << i
                                << " (Epoch " << epoch << ") due to error threshold.\n";
                                break;
                            }
                        }
                    }
                    
                    cout << "Epoch " << epoch << " completed. Average Loss: "
                    << epoch_loss / total_samples << endl;
                }
                
                print();
                auto stop = high_resolution_clock::now();
                auto duration = duration_cast<microseconds>(stop - start);
                cout << "\nTraining completed in " << duration.count() << " microseconds." << endl;
    }

    
    void print() {
        // Print table header
        cout << std::setw(7) << "Layer"
        << std::setw(10) << "Neuron"
        << std::setw(13) << "Output"
        << std::setw(20) << "Activated Output"
        << std::setw(10) << "Bias"
        << std::setw(20) << "Weights" << endl;
        
        cout << std::string(80, '-') << endl; // Separator line
        
        // Iterate through each layer
        for (size_t i = 0; i < layers.size(); ++i) {
            // Iterate through each neuron in the layer
            for (size_t j = 0; j < num_neurons[i]; ++j) {
                // Print layer index, neuron index, output, activated output, and bias
                cout << std::setw(5) << i
                << setw(10) << j
                << setw(15) << setprecision(6) << layers[i]->get_output()[j]
                << setw(15) << setprecision(6) << layers[i]->get_activated_output()[j]
                << setw(17) << setprecision(6) << layers[i]->get_bias()[j]
                << setw(15);
                
                // Print weights for the neuron
                for (size_t k = 0; k < layers[i]->get_weight()[j].size(); ++k) {
                    cout << std::setprecision(6) << layers[i]->get_weight()[j][k];
                    if (k < layers[i]->get_weight()[j].size() - 1) {
                        cout << ", "; // Separate weights with commas
                    }
                }
                cout << endl; // Newline for the next neuron
            }
        }
        // Print probabilities for the last layer
        cout << endl << "PROBABILITY: ";
        for (auto& prob : layers.back()->get_probability()) {
            cout << std::setprecision(6) << prob << ", ";
        }
        cout << endl << "LOSS: " << layers.back() -> get_loss();
        cout << endl << endl;
    }

    void print_output(){
        // Print probabilities for the last layer
        cout << endl << "PROBABILITY: ";
        for (auto& prob : layers.back()->get_probability()) {
            cout << std::setprecision(6) << prob << ", ";
        }
        cout << endl << "LOSS: " << layers.back() -> get_loss();
        cout << endl << endl;
    }

    const vector<unique_ptr<Layer>>& get_layers() const noexcept {
        return layers;
    }

    void set_dropout(vector <double>& keep_prob) {
        assert(keep_prob.size() == layers.size());
        for(size_t i = 0; i < layers.size(); ++i) {
            layers[i]->set_dropout(keep_prob[i]);
        }
    }

    vector<double> get_output() {
        return layers.back()->get_output();
    }

    vector<double> get_activated_output(){
        return layers.back() -> get_activated_output();
    }

    vector<double> get_error() {
        return layers.back()->get_error();
    }

    vector<vector<double>> get_weight() {
        return layers.back()->get_weight();
    }

    vector<double> get_bias() {
        return layers.back()->get_bias();
    }
    
    vector <double> get_gradient(){
        return layers.back()->get_gradient();
    }

    vector <double> get_passing_gradient(){
        return layers.front() -> get_gradient();
    }

    double get_loss(){
        return layers.back() -> get_loss();
    }

    // Set step size for all layers
    void set_step_size(int stepsize) noexcept {
        for (auto& layer : layers) {
            layer->set_step_size(stepsize);
        }
    }

    void set_target(vector<double>& targets) {
        if (!layers.empty()) {
            layers.back()->set_target(targets);
        }
    }

    void set_softlabeling(vector<double>& labels, double resolution) {
        if (!layers.empty()) {
            layers.back()->set_softlabeling(labels, resolution);
        }
    }

    void set_hardlabeling(vector<double>& labels) {
        if (!layers.empty()) {
            layers.back()->set_hardlabeling(labels);
        }
    }

    void set_input(vector<double>& inputs) {
        if (!layers.empty()) {
            layers.front()->set_input(inputs);
        }
    }

    void set_bias(vector<double>& biases) {
        if (!layers.empty()) {
        }
    }

    void set_weight(vector<vector<double>>& weights) {
        if (!layers.empty()) {
            layers.front()->set_weight(weights);
        }
    }

    void set_error(vector<double>& error) {
        if (!layers.empty()) {
            layers.back()->set_error(error);
        }
    }

    void set_gradient(vector<double>& gradients) {
        if (!layers.empty()) {
            layers.back()->set_gradient(gradients);
        }
    }

    void set_passing_gradient(vector<double>& gradients){
        layers.back()->set_passing_gradient(gradients);
    }

    // --- Model Persistence ---
    void save_model(const std::string& filename) {
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs) throw std::runtime_error("Failed to open file for saving model");
        size_t n_layers = layers.size();
        ofs.write(reinterpret_cast<const char*>(&n_layers), sizeof(n_layers));
        for (const auto& layer : layers) {
            size_t n_neurons = layer->get_neuron().size();
            ofs.write(reinterpret_cast<const char*>(&n_neurons), sizeof(n_neurons));
            for (const auto& neuron : layer->get_neuron()) {
                const auto& w = neuron->get_weight();
                double b = neuron->get_bias();
                size_t ws = w.size();
                ofs.write(reinterpret_cast<const char*>(&ws), sizeof(ws));
                ofs.write(reinterpret_cast<const char*>(w.data()), ws * sizeof(double));
                ofs.write(reinterpret_cast<const char*>(&b), sizeof(double));
            }
        }
        // Optionally: save hyperparameters, architecture, etc.
    }
    void load_model(const std::string& filename) {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs) throw std::runtime_error("Failed to open file for loading model");
        size_t n_layers = 0;
        ifs.read(reinterpret_cast<char*>(&n_layers), sizeof(n_layers));
        if (n_layers != layers.size()) throw std::runtime_error("Model architecture mismatch");
        for (auto& layer : layers) {
            size_t n_neurons = 0;
            ifs.read(reinterpret_cast<char*>(&n_neurons), sizeof(n_neurons));
            if (n_neurons != layer->get_neuron().size()) throw std::runtime_error("Layer size mismatch");
            for (const auto& neuron : layer->get_neuron()) {
                size_t ws = 0;
                ifs.read(reinterpret_cast<char*>(&ws), sizeof(ws));
                std::vector<double> w(ws);
                ifs.read(reinterpret_cast<char*>(w.data()), ws * sizeof(double));
                double b = 0;
                ifs.read(reinterpret_cast<char*>(&b), sizeof(double));
                neuron->set_weight(w);
                neuron->set_bias(b);
            }
        }
    }

    // --- Fit/Train with optional validation and callbacks ---
    void fit(vector<vector<double>>& X_train,
             vector<vector<double>>& y_train,
             int epochs, int batch_size = 1,
             vector<vector<double>>* X_val = nullptr,
             vector<vector<double>>* y_val = nullptr,
             double threshold = 0.05) {
                
                size_t n_samples = X_train.size();
                train_loss_history.clear();
                val_loss_history.clear();
                
                for (int epoch = 1; epoch <= epochs; ++epoch) {
                    double epoch_loss = 0.0;
                    
                    for (size_t i = 0; i < n_samples; i += batch_size) {
                        size_t end = min(i + batch_size, n_samples);
                        for (size_t j = i; j < end; ++j) {
                            set_input(const_cast<vector<double>&>(X_train[j]));
                            set_target(const_cast<vector<double>&>(y_train[j]));
                            feedforward();
                            probability();
                            epoch_loss += get_loss();
                            backpropagation();
                            regularization();
                        }
                    }
                    train_loss_history.push_back(epoch_loss / n_samples);
                    // Validation
                    double val_loss = 0.0;
                    
                    if (X_val && y_val) {
                        for (size_t i = 0; i < X_val->size(); ++i) {
                            set_input(const_cast<std::vector<double>&>((*X_val)[i]));
                            set_target(const_cast<std::vector<double>&>((*y_val)[i]));
                            feedforward();
                            probability();
                            val_loss += get_loss();
                        }
                        val_loss /= X_val->size();
                        val_loss_history.push_back(val_loss);
                    }
                    if (epoch_callback) epoch_callback(epoch, train_loss_history.back(), val_loss_history.empty() ? 0.0 : val_loss_history.back());
                }
            }
            // --- Metrics and Callbacks ---
            void set_epoch_callback(std::function<void(int, double, double)> cb) { epoch_callback = cb; }
            vector<double>& get_train_loss_history() { return train_loss_history; }
            vector<double>& get_val_loss_history() { return val_loss_history; }
};

#endif