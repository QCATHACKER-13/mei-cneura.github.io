#ifndef BENCHMARKTEST_H
#define BENCHMARKTEST_H

#include <chrono>
#include <fstream>
#include <iomanip>
#include "cneura.h"

void benchmark_training(Neural& net, const vector<double>& targets, int epochs, double error_margin, const vector<double>& labels) {
    using namespace std::chrono;

    ofstream logfile("benchmark_results.csv");
    logfile << "Epoch,TotalError,LabelError,FinalOutput\n";

    auto start_time = high_resolution_clock::now();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        net.set_target(targets, 1e-6);
        net.set_hardlabeling(labels); // Set hard labeling for the network
        net.feedforward();
        net.probability_calculation();
        net.backpropagation();

        // Calculate total error
        double total_error = 0.0;
        for (const auto& layer : net.get_layers()) {
            for (double e : layer->get_error()) {
                total_error += abs(e); // Accumulate absolute errors
            }
        }

        // Calculate label-specific error
        double label_error = 0.0;
        const auto& outputs = net.get_output();
        for (size_t i = 0; i < labels.size(); ++i) {
            label_error += abs(labels[i] - outputs[i]); // Compare labels with outputs
        }

        // Log data
        logfile << epoch << "," << total_error << "," << label_error << ",";
        for (double out : outputs) {
            logfile << out << " ";
        }
        logfile << "\n";

        // Check for label-specific error threshold
        if (label_error > error_margin) {
            cout << "Warning: High label error at epoch " << epoch << " with label error: " << label_error << endl;
        }

        // Check for convergence
        if (total_error < error_margin) {
            cout << "Converged at epoch " << epoch << " with total error: " << total_error << endl;
            break;
        }
    }

    auto end_time = high_resolution_clock::now();
    duration<double> duration_sec = end_time - start_time;
    cout << "Training completed in " << duration_sec.count() << " seconds." << endl;
}

#endif // BENCHMARKTEST_H