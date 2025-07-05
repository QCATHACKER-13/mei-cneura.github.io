/* Project: NEURA 
NEURA is an artificial neuron and network development research project 
focused on artificial intelligence for data analysis. The goal is to find 
the optimal neural network structure and improve artificial neurons for 
efficiency and speed.

Project Development and Innovations:
- Project Neura: Neuron investigation project
- Project Mei: Neural Net Architecture investigation project

Developer: Christopher Emmanuelle J. Visperas, Applied Physics Researcher*/


#ifndef CTRANSFORMER_H
#define CTRANSFORMER_H

#pragma once

#include <iostream>
#include <memory>
#include "../transformer_model/transformer_neural_config.h"
#include "../transformer_model/transformer_parameter_config.h"
#include "../core_model/cneura.h"

using namespace std;


class Transformer{
    private:
        Hyperparameters hyperpara_config;
        DataConfig data_config;
        NeuralConfig neural_config;

        unique_ptr<Neural> encoder, decoder;


    public:
        Transformer() {
            // Initialize encoder
            encoder = make_unique<Neural>(
                hyperpara_config.num_neuron_encoder,
                data_config.normalized_inputs,
                hyperpara_config.learning_rate,
                hyperpara_config.decay_rate,
                hyperpara_config.beta,
                neural_config.initializer,
                neural_config.distribution,
                neural_config.actFunc,
                neural_config.lr_schedule,
                neural_config.opt,
                neural_config.lossFunc
            );

            encoder -> set_step_size(hyperpara_config.step_size);
            encoder -> set_dropout(data_config.keep_prob);

            encoder -> layer_initialization();
            encoder -> feedforward();
            encoder -> probability();
            vector<double> encoder_output = encoder -> get_activated_output();
            // Initialize decoder
            decoder = make_unique<Neural>(
                hyperpara_config.num_neuron_decoder,
                encoder_output,
                hyperpara_config.learning_rate,
                hyperpara_config.decay_rate,
                hyperpara_config.beta,
                neural_config.initializer,
                neural_config.distribution,
                neural_config.actFunc,
                neural_config.lr_schedule,
                neural_config.opt,
                neural_config.lossFunc
            );
            decoder -> set_step_size(hyperpara_config.step_size);
            decoder -> set_dropout(data_config.keep_prob);
            decoder -> set_target(data_config.decoder_target);
            //decoder -> set_hardlabeling(data_config.label);
            decoder -> gaussian_labelling(data_config.decoder_target, data_config.labeled_data);

            decoder -> layer_initialization();
        }

        void initialize() {
            encoder -> layer_initialization();
            decoder -> layer_initialization();
        }

        void feedforward(){
            encoder -> feedforward();
            vector<double> encoder_output = encoder -> get_activated_output();
            encoder -> probability();
            decoder -> set_input(encoder_output);
            decoder -> feedforward();
            decoder -> probability();
        }

        void backpropagation(){
            decoder -> backpropagation();
            decoder -> regularization();
            vector<double> decoder_passing_gradient = decoder -> get_passing_gradient();
            encoder -> set_passing_gradient(decoder_passing_gradient);
            encoder -> backpropagation();
            encoder -> regularization();
        }

        void learning(int steps, double threshold = 0.05) {
            auto start = high_resolution_clock::now();
            
            double epoch_loss = 0.0;

            initialize();
            
            #pragma omp parallel for
            for (int step = 1; step <= steps; ++step) {
                
                feedforward();
                
                const auto& errors = decoder->get_error();
                
                bool below_threshold = all_of(errors.begin(), errors.end(), [&](double err) {
                    return std::abs(err) < threshold;
                });
                
                epoch_loss += decoder -> get_loss();
                
                if (below_threshold) {
                    break;
                }
                backpropagation();
            }
            
            encoder -> print();
            decoder -> print();
            
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);
            
            cout << "Final Loss: " << decoder -> get_loss() << endl;
            cout << "Average Loss: " << epoch_loss/steps << endl;
            cout << "Training completed in " << duration.count() << " microseconds" << endl;
        }
        
        void enforce_learning(int epochs, double threshold = 0.05) {
            auto start = high_resolution_clock::now();
            double epoch_loss = 0.0;
            
            #pragma omp parallel for
            for (int epoch = 1; epoch <= epochs; ++epoch) {
                initialize();
                for (int step = 1; step <= epoch; ++step) {
                    feedforward();
                    
                    const auto& errors = decoder -> get_error();
                    
                    bool below_threshold = all_of(errors.begin(), errors.end(), [&](double err) {
                        return std::abs(err) < threshold;
                    });
                    
                    double current_loss = decoder -> get_loss();
                    
                    epoch_loss += current_loss;

                    //encoder -> print();
                    //decoder -> print();
                    //encoder -> print_output();
                    //decoder -> print_output();
                    
                    if (below_threshold) {
                        //encoder -> print();
                        //decoder -> print();
                        //encoder -> print_output();
                        //decoder -> print_output();
                        cout<<"Early stop of epoch "<<epoch<<" with step " <<step << endl;
                        break;
                    }
                    backpropagation();
                }
                //encoder -> print();
                //decoder -> print();
                //encoder -> print_output();
                //decoder -> print_output();
                epoch_loss /= epoch;
            }
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);
            
            std::cout << "Final Loss: " << decoder->get_loss() << endl;
            std::cout << "Average Loss: " << epoch_loss << endl;
            cout << "Training completed in " << duration.count() << " microseconds" << endl;
        }
};

#endif // CTRANSFORMER_H