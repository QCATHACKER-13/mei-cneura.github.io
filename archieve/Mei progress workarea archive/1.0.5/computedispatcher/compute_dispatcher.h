/* Project: NEURA 
NEURA is an artificial neuron and network development research project 
focused on artificial intelligence for data analysis. The goal is to find 
the optimal neural network structure and improve artificial neurons for 
efficiency and speed.

Project Development and Innovations:
- Project Neura: Neuron investigation project
- Project Mei: Neural Net Architecture investigation project

Developer: Christopher Emmanuelle J. Visperas, Applied Physics Researcher*/
#ifndef COMPUTE_DISPATCHER_H
#define COMPUTE_DISPATCHER_H

#include <memory>
#include <vector>
#include <iostream>
#include <stdexcept>

// Forward declaration
class Neuron;

enum class ComputeTarget {
    CPU,
    GPU,
    HYBRID
};

class ComputeDispatcher {
public:
    virtual void process_neurons(std::vector<std::unique_ptr<Neuron>>& neurons, const std::vector<float>& input) = 0;
    virtual ~ComputeDispatcher() = default;
};

class CPUDispatcher : public ComputeDispatcher {
public:
    void process_neurons(std::vector<std::unique_ptr<Neuron>>& neurons, const std::vector<float>& input) override;
};

// Stub for future GPU support
class GPUDispatcher : public ComputeDispatcher {
public:
    void process_neurons(std::vector<std::unique_ptr<Neuron>>& neurons, const std::vector<float>& input) override {
        std::cerr << "[GPU] Not implemented yet.\n";
    }
};

// Declaration only! Implementation must be in a .cpp file.
std::unique_ptr<ComputeDispatcher> CreateDispatcher(ComputeTarget& selected_target);

#endif // COMPUTE_DISPATCHER_H