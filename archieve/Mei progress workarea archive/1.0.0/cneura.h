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
#include <memory> // For std::unique_ptr
#include <algorithm>
#include <cmath>
#include <numeric>
#include "clayer.h"

using namespace std;

enum lrmode {SUPERVISED, UNSUPERVISED, REINFORCEMENT, SELF_LEARNING, MULTI_TASK}; // Enum for learning modes

class Neural{
    private:
        vector<unique_ptr<Layer>> layers; // Use unique_ptr for better memory management
        vector<vector<vector<double>>> weight; // Stores weights for all layers

    public:
    Neural(const vector<size_t>& num_neurons, const vector<double>& inputs,
        const double& learning_rate, const double& decay_rate, const vector<double>& beta,
        actfunc actFunc, lrs lr_schedule, optimizer opt, lossfunc lossFunc) {
            
            if (inputs.empty()) {
                throw invalid_argument("Input vector cannot be empty.");
            }
            
            layers.reserve(num_neurons.size());
            vector<double> current_input = inputs;
            
            for (size_t i = 0; i < num_neurons.size(); ++i) {
                neurontype layer_type = (i == num_neurons.size() - 1) ? OUTPUT : HIDDEN;
                
                auto layer = make_unique<Layer>(
                    num_neurons[i], current_input,
                    learning_rate, decay_rate, beta,
                    layer_type, actFunc, lr_schedule, opt, lossFunc
                );
                
                if (!layer) {
                    throw runtime_error("Failed to initialize layer " + to_string(i));
                }
                
                layers.emplace_back(move(layer));
                current_input = layers.back()->get_activated_output();
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
                vector<double> current_input = layers.front()->get_input(); // Start with the input of the first layer
            
                for (size_t i = 0; i < layers.size(); ++i) {
                    layers[i]->set_input(current_input); // Set the input for the current layer
                    layers[i]->feedforward();           // Perform feedforward computation
                    current_input = layers[i]->get_activated_output(); // Update the input for the next layer
                }
            }

            void probability_calculation() {
                if (layers.empty()) return;
            
                // Perform probability calculation for the output layer
                layers.back()->softmax();
                layers.back()->loss_function();
                layers.back()->loss_derivative();

                //Propagate errors for hidden layers
                for (int i = static_cast<int>(layers.size()) - 2; i > -1; i--) {
                    layers[i]->set_probability(layers[i + 1]->get_probability());
                }
            }
            
            void backpropagation() {
                if (layers.empty()) return;
            
                // Backpropagation for the output layer
                layers.back()->backpropagation();
            
                // Backpropagation for hidden layers
                for (int i = static_cast<int>(layers.size()) - 2; i > -1; --i) {
                    // Set the error for the current hidden layer based on the next layer's error
                    layers[i]->set_error(layers[i + 1]->get_error());
            
                    // Perform backpropagation for the current hidden layer
                    layers[i]->backpropagation();
                }
            }

            void learning(int step_size,
                const vector<double>& training_targets, 
                const vector<double>& labeling, string label_type = "hardlabeling"
            ) {
                if (layers.empty()) {
                    cerr << "Error: Neural network has no layers initialized." << endl;
                        return;
                    }
                    set_target(training_targets);
                    if (label_type == "softlabeling") set_softlabeling(labeling, 1e-2);
                    if (label_type == "hardlabeling") set_hardlabeling(labeling);

                    double total_loss = 0.0;
                    int correct_predictions = 0;
                    
                    for (int step = 1; step <= step_size; step++) {
                        cout << "Step " << step << "/" << step_size << endl;
                        
                        // Perform feedforward
                        feedforward();
                        probability_calculation();

                        // Check if the predicted class matches the target class
                        int predicted_class = distance(layers.back() ->get_activated_output().begin(), 
                        max_element(layers.back() ->get_activated_output().begin(), 
                        layers.back() ->get_activated_output().end()));
                        
                        int target_class = distance(training_targets.begin(), max_element(training_targets.begin(), training_targets.end()));
                        
                        if (predicted_class == target_class) {
                            correct_predictions++;
                        }

                        // Calculate loss and accumulate it
                        cout << "Loss: " << layers.back()->get_loss() << endl;
                        
                        // Perform backpropagation
                        backpropagation();
                    }
                    // Optional: Add early stopping or learning rate adjustment here
                    cout << "Learning completed!" << endl;
                }


            void train(const vector<double>& training_inputs, 
                const vector<double>& training_targets, 
                int epochs, 
                int batch_size) {
                    if (layers.empty()) {
                        cerr << "Error: Neural network has no layers initialized." << endl;
                        return;
                    }
                    
                    // Ensure inputs and targets are valid
                    if (training_inputs.size() != training_targets.size()) {
                        throw invalid_argument("Training inputs and targets must have the same size.");
                    }

                    set_input(training_inputs);
                    set_target(training_targets);
                    
                    for (int epoch = 1; epoch <= epochs; ++epoch) {
                        cout << "Epoch " << epoch << "/" << epochs << endl;
                        
                        double total_loss = 0.0;
                        int num_batches = (training_inputs.size() + batch_size - 1) / batch_size;
                        
                        for (int batch = 0; batch < num_batches; ++batch) {
                            // Process each batch
                            int start_idx = batch * batch_size;
                            int end_idx = min(start_idx + batch_size, static_cast<int>(training_inputs.size()));
                            
                            for (int i = start_idx; i < end_idx; ++i) {
                                // Set input and target for the current sample
                                
                                // Perform feedforward
                                feedforward();
                                probability_calculation();

                                // Calculate loss and accumulate it
                                total_loss += layers.back()->get_loss();
                                
                                // Perform backpropagation
                                backpropagation();
                            }
                        }
                        // Display average loss for the epoch
                        cout << "Average Loss: " << total_loss / training_inputs.size() << endl;
                        
                        // Optional: Add early stopping or learning rate adjustment here
                        }
                        cout << "Training completed!" << endl;
                    }

            void debug_feedforward() const {
                for (size_t i = 0; i < layers.size(); ++i) {
                    cout << "Layer " << i + 1 << " Outputs: ";
                    for (const auto& val : layers[i]->get_activated_output()) {
                        cout << val << " ";
                    }
                    cout << endl;
                }
            }

            void debug_weight() const {
                for (size_t i = 0; i < layers.size(); ++i) {
                    cout << "Layer " << i + 1 << " Weights: ";
                    for (const auto& row : layers[i]->get_weight()) {
                        for (const auto& val : row) {
                            cout << val << " ";
                        }
                        cout << endl;
                    }
                }
            }
            void debug_bias() const {
                for (size_t i = 0; i < layers.size(); ++i) {
                    cout << "Layer " << i + 1 << " Biases: ";
                    for (const auto& val : layers[i]->get_bias()) {
                        cout << val << " ";
                    }
                    cout << endl;
                }
            }

            void debug_probability() const {
                for (size_t i = 0; i < layers.size(); ++i) {
                    cout << "Layer " << i + 1 << " Probabilities: ";
                    for (const auto& val : layers[i]->get_probability()) {
                        cout << val << " ";
                    }
                    cout << endl;
                }
            }

            void debug_error() const {
                for(size_t i = 0; i < layers.size(); ++i) {
                    cout << "Layer " << i + 1 << " Error: ";
                    for (const auto& val : layers[i]->get_error()) {
                        cout << val << " ";
                    }
                    cout << endl;
                }
            }

            void debug_loss() const {
                cout << "Loss: ";
                auto loss = layers.back()->get_loss();
                cout << loss << endl;
            }
            
            void print() const {
                for (const auto& layer : layers) {
                    layer->debug_state();
                }
            }

            const vector<unique_ptr<Layer>>& get_layers() const noexcept {
                return layers;
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