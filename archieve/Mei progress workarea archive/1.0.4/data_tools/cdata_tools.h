/*

This cdata++.h is a header file for statistics, mathematical methods 
formula for computational data science. It is tool for data processing, 
filtering and categorize selection, and other formula tools for data science
and data analytics. And also used on the Project NEURA for data training.


Project NEURA is an artificial neuron and network development research project 
focused on artificial intelligence for data analysis. The goal is to find 
the optimal neural network structure and improve artificial neurons for 
efficiency and speed.

Future projects:
- Project Mei: Neuron investigation project
- Project Raiden: Network development 

Developer: Christopher Emmanuelle J. Visperas, Applied Physics Researcher
*/

#ifndef CDATA_TOOLS_H
#define CDATA_TOOLS_H

#pragma once

#include <iostream>
#include <cmath>     // for exp()
#include <vector>
#include <algorithm> // for max_element, accumulate
#include <numeric>   // for accumulate
#include <cassert>

using namespace std;

enum NORMTYPE { MINMAX, SYMMETRIC, MEANCENTER, ZSCORE }; // Enum for normalization types

class Data{
    private:
        vector<double> input, target;
        NORMTYPE ntype_norm; // Normalization type

    public:
        Data(vector<double> inputs) : input(inputs){
            //Intialize the constructor
            this -> input = inputs;
        }

        vector<double> dataset_normalization(NORMTYPE ntype_norm) {
            this -> ntype_norm = ntype_norm;
            double min = *min_element(input.begin(), input.end());
            double max = *max_element(input.begin(), input.end());

            double range = max - min;
            assert(range > 0); // Ensure valid range

            double mean = accumulate(input.begin(), input.end(), 0.0) / input.size();
            double stddev = sqrt(accumulate(input.begin(), input.end(), 0.0, 
                [mean](double sum, double val) { return sum + (val - mean) * (val - mean); }) / input.size());
            assert(stddev > 0); // Ensure valid standard deviation

            for(size_t i = 0; i < input.size(); i++){
                assert(i < input.size());
                
                switch(ntype_norm) {
                    case MINMAX:
                        // Ensure index is within limits
                        input[i] = ((input[i] - min)/range);
                        break;
                        
                    case SYMMETRIC:
                        // Implement symmetric normalization
                        input[i] = (2 * ((input[i] - min) / range)) - 1;
                        break;
                        
                    case MEANCENTER:
                        // Implement mean centering
                        input[i] = (2 * ((input[i] - mean) / range)) - 1;
                        break;
                        
                    case ZSCORE:
                        // Implement z-score normalization
                        input[i] = (input[i] - mean) / stddev;
                        break;
                }
            }
            return input;
        }

        vector<double> targetdataset_normalization(NORMTYPE ntype_norm, vector<double> target) {
            this -> target = target;
            this -> ntype_norm = ntype_norm;
            double min = *min_element(input.begin(), input.end());
            double max = *max_element(input.begin(), input.end());

            double range = max - min;
            assert(range > 0); // Ensure valid range

            double mean = accumulate(input.begin(), input.end(), 0.0) / input.size();
            double stddev = sqrt(accumulate(input.begin(), input.end(), 0.0, 
                [mean](double sum, double val) { return sum + (val - mean) * (val - mean); }) / input.size());
            assert(stddev > 0); // Ensure valid standard deviation

            for(size_t i = 0; i < target.size(); i++){
                assert(i < target.size());
                
                switch(ntype_norm) {
                    case MINMAX:
                        // Ensure index is within limits
                        target[i] = ((target[i] - min)/range);
                        break;
                        
                    case SYMMETRIC:
                        // Implement symmetric normalization
                        target[i] = (2 * ((target[i] - min) / range)) - 1;
                        break;
                        
                    case MEANCENTER:
                        // Implement mean centering
                        target[i] = (2 * ((target[i] - mean) / range)) - 1;
                        break;
                        
                    case ZSCORE:
                        // Implement z-score normalization
                        target[i] = (target[i] - mean) / stddev;
                        break;
                }
            }
            return target;
        }

        double targetdata_normalization(NORMTYPE ntype_norm, double target_data) {
            this -> ntype_norm = ntype_norm;
            double target_value = target_data;
            double min = *min_element(input.begin(), input.end());
            double max = *max_element(input.begin(), input.end());

            double range = max - min;
            assert(range > 0); // Ensure valid range

            double mean = accumulate(input.begin(), input.end(), 0.0) / input.size();
            double stddev = sqrt(accumulate(input.begin(), input.end(), 0.0, 
                [mean](double sum, double val) { return sum + (val - mean) * (val - mean); }) / input.size());
            assert(stddev > 0); // Ensure valid standard deviation

            switch(ntype_norm) {
                    case MINMAX:
                        // Ensure index is within limits
                        target_value = ((target_value - min)/range);
                        break;
                        
                    case SYMMETRIC:
                        // Implement symmetric normalization
                        target_value = (2 * ((target_value - min) / range)) - 1;
                        break;
                        
                    case MEANCENTER:
                        // Implement mean centering
                        target_value = (2 * ((target_value - mean) / range)) - 1;
                        break;
                        
                    case ZSCORE:
                        // Implement z-score normalization
                        target_value = (target_value - mean) / stddev;
                        break;
            }
            return target_value;
        }
};

#endif