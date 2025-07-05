/* Project: NEURA 
NEURA is an artificial neuron and network development research project 
focused on artificial intelligence for data analysis. The goal is to find 
the optimal neural network structure and improve artificial neurons for 
efficiency and speed.

Project Development and Innovations:
- Project Neura: Neuron investigation project
- Project Mei: Neural Net Architecture investigation project

Developer: Christopher Emmanuelle J. Visperas, Applied Physics Researcher

*/

#ifndef CNEURA_H
#define CNEURA_H

#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <chrono>
#include "clayer.h"

using std::vector;
using std::unique_ptr;
using std::make_unique;
using std::invalid_argument;
using std::runtime_error;
using std::cout;
using std::endl;
using std::min;
using std::max_element;
using std::distance;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::duration;

enum lrmode {SUPERVISED, UNSUPERVISED, REINFORCEMENT, SELF_LEARNING, MULTI_TASK};

class Neural {
private:
    vector<unique_ptr<Layer>> layers;
    vector<size_t> num_neurons;

public:
    Neural(const vector<size_t>& num_neurons, const vector<double>& inputs,
           const double& learning_rate, const double& decay_rate, const vector<double>& beta,
           ACTFUNC actFunc, LEARNRATE lr_schedule, OPTIMIZER opt, LOSSFUNC lossFunc) : num_neurons(num_neurons) {
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
                layer_type, actFunc, lr_schedule, opt, lossFunc
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

    // Set step size for all layers
    void set_step_size(int stepsize) noexcept {
        for (auto& layer : layers) {
            layer->set_step_size(stepsize);
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
            current_input = layers[i]->get_activated_output();
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
        layers.back()->backpropagation();

        #pragma omp parallel for
        for (int i = static_cast<int>(layers.size()) - 2; i >= 0; --i) {
            layers[i]->set_error(layers[i + 1]->get_error());
            layers[i]->backpropagation();
        }
    }

    void learning(int steps, double threshold = 0.05) {
        if (layers.empty()) throw invalid_argument("The layers didn't initialize properly.");
        
        int count = 0;

        auto start = high_resolution_clock::now();

        #pragma omp parallel for
        for (int step = 1; step <= steps; ++step) {
            feedforward();
            probability();

            //print();
            for(auto& error : layers.back()->get_error()){
                if(error < threshold){
                    count++;
                }
            }
            if(count == layers.back()->get_error().size()){
                print();
                cout << "Learning completed!" << endl;
                cout << "Step " << step << "/" << steps << endl;
                cout << "Error Counts: "<< count << ", Loss: " << layers.back()->get_loss() << endl;
                break;
            }
            regularization();
            backpropagation();
            count = 0;
        }
        cout << "Error Counts: "<< count << ", Loss: " << layers.back()->get_loss() << endl;
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        cout << "Training completed!" << endl;
        cout << " | Time taken: " << duration.count() << " microseconds" << endl;
    }

    void train(const vector<double>& training_inputs,
               const vector<double>& training_targets,
               int epochs,
               int batch_size) {
        if (layers.empty()) {
            throw runtime_error("Neural network has no layers initialized.");
        }
        if (training_inputs.empty() || training_targets.empty()) {
            throw invalid_argument("Training inputs and targets must not be empty.");
        }
        // Note: In most cases, training_inputs.size() != training_targets.size()
        // unless it's a 1:1 mapping (e.g., autoencoder). Adjust as needed for your use case.

        //set_input(training_inputs);
        //set_target(training_targets);

        for (int epoch = 1; epoch <= epochs; ++epoch) {
            cout << "Epoch " << epoch << "/" << epochs << endl;
            double total_loss = 0.0;
            int num_batches = (training_inputs.size() + batch_size - 1) / batch_size;

            for (int batch = 0; batch < num_batches; ++batch) {
                int start_idx = batch * batch_size;
                int end_idx = min(start_idx + batch_size, static_cast<int>(training_inputs.size()));

                for (int i = start_idx; i < end_idx; ++i) {
                    // For a real batch, you should slice the input and target vectors here.
                    // This code assumes single-sample batches.
                    feedforward();
                    probability();
                    //Print Layers Output and others parameters
                    print();
                    
                    total_loss += layers.back()->get_loss();
                    //regularization();
                    backpropagation();
                }
            }
            cout << "Average Loss: " << total_loss / training_inputs.size() << endl;
        }
        
    }
    
    void print() const {
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
                << std::setw(10) << j
                << std::setw(15) << std::setprecision(6) << layers[i]->get_output()[j]
                << std::setw(15) << std::setprecision(6) << layers[i]->get_activated_output()[j]
                << std::setw(17) << std::setprecision(6) << layers[i]->get_bias()[j]
                << std::setw(15);
                
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

    vector<double> get_output() const {
        return layers.back()->get_output();
    }

    vector<double> get_error() const {
        return layers.back()->get_error();
    }

    vector<vector<double>> get_weight() const {
        return layers.back()->get_weight();
    }

    vector<double> get_bias() const {
        return layers.back()->get_bias();
    }

    void set_target(const vector<double>& targets) {
        if (!layers.empty()) {
            layers.back()->set_target(targets);
        }
    }

    void set_softlabeling(const vector<double>& labels, double resolution) {
        if (!layers.empty()) {
            layers.back()->set_softlabeling(labels, resolution);
        }
    }

    void set_hardlabeling(const vector<double>& labels) {
        if (!layers.empty()) {
            layers.back()->set_hardlabeling(labels);
        }
    }

    void set_input(const vector<double>& inputs) {
        if (!layers.empty()) {
            layers.front()->set_input(inputs);
        }
    }

    void set_bias(const vector<double>& biases) {
        if (!layers.empty()) {
            layers.front()->set_bias(biases);
        }
    }

    void set_weight(const vector<vector<double>>& weights) {
        if (!layers.empty()) {
            layers.front()->set_weight(weights);
        }
    }

    void set_error(const vector<double>& error) {
        if (!layers.empty()) {
            layers.back()->set_error(error);
        }
    }
};

#endif