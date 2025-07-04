/* Project: MEI 
NEURA is an artificial neuron and network development research project 
focused on artificial intelligence for data analysis. The goal is to find 
the optimal neural network structure and improve artificial neurons for 
efficiency and speed.

Project Development and Innovations:
- Project Neura: Neuron investigation project
- Project Mei: Neural Net Architecture investigation project
- Project Raiden: Network development 
- Hardware integration(Raiden Mei Project): After successfully developing the artificial 
  neuron network, the project will transition into hardware implementation.

Developer: Christopher Emmanuelle J. Visperas, Applied Physics Researcher
*/

#ifndef MEI_H
#define MEI_H


#include "cneuron.h"
#include "clayer.h"
#include "cneura.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

class MEI{
    private:
        Neural LeftWing, RightWing, Thalamus;
        size_t num_neurons;
        vector<double> inputs, target, labeling;
        double resolution;

    public:
        MEI(const vector<vector<size_t>>& num_neurons, const vector<double>& inputs,
            const double& learning_rate, const double& decay_rate, const vector<double>& beta,
            normtype ntype_norm, actfunc actFunc, lrs lr_schedule, optimizer opt, lossfunc lossFunc)
            : LeftWing(num_neurons[0], inputs, learning_rate, decay_rate, beta, ntype_norm, actFunc, lr_schedule, opt, lossFunc),
              RightWing(num_neurons[1], inputs, learning_rate, decay_rate, beta, ntype_norm, actFunc, lr_schedule, opt, lossFunc),
              Thalamus(num_neurons[2], inputs, learning_rate, decay_rate, beta, ntype_norm, actFunc, lr_schedule, opt, lossFunc) {
                if (inputs.empty()) {
                    throw invalid_argument("Input vector cannot be empty.");
                }
            }

        void feedforward() {
            LeftWing.feedforward();
            RightWing.feedforward();
            Thalamus.feedforward();
        }
        
        void backpropagation() {
            LeftWing.backpropagation();
            RightWing.backpropagation();
            Thalamus.backpropagation();
        }

        void set_target(const vector<double>& targets) {
            target = targets;
        }

        void set_labeling(const vector<double>& labels) {
            labeling = labels;
        }

        void set_resolution(double res) {
            resolution = res;
        }
};

#endif