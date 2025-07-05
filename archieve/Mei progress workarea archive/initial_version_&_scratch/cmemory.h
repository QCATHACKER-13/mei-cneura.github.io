/* Project: NEURA 
NEURA is an artificial neuron and network development research project 
focused on artificial intelligence for data analysis. The goal is to find 
the optimal neural network structure and improve artificial neurons for 
efficiency and speed.

Future projects:
- Project Mei: Neuron investigation project

Developer: Christopher Emmanuelle J. Visperas, Applied Physics Researcher*/


#ifndef CMEMORY_H
#define CMEMORY_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

class Memory{
    private: 
        vector<double> weight;
        double bias;
    
    public:
        void save_memory(const string& filename){
            ofstream file(filename);
            if(file.is_open()){
                for (const auto& w : weight) file << w << " ";
                file << bias;
                file.close();
            }
        }

        void load_memory(const string& filename){
            ifstream file(filename);
            if(file.is_open()){
                for (auto& w : weight) file >> w;
                file >> bias;
                file.close();
            }
        }

        void save_weights(vector<double> weights){weight = weights;}
        void save_bias(double biases){bias = biases;}

};

#endif