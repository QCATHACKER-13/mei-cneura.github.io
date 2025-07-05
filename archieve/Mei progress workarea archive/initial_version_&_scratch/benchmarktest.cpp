#include <iostream>
#include <chrono>
#include <vector>
#include "clayer.h"
#include "cneura.h"
#include "cdata_tools.h"

using namespace std;
using namespace std::chrono;

// Constants
const double ERROR_THRESHOLD = 0.05; // Error threshold for convergence
const int MAX_COUNT = 9;             // Number of errors to check for convergence

// Function to count errors below the threshold
int count_errors_below_threshold(const vector<double>& errors, double threshold) {
    int count = 0;
    for (const auto& error : errors) {
        if (abs(error) < threshold) {
            count++;
        }
    }
    return count;
}

int main() {
    // Define the number of neurons on each layer
    vector<size_t> num_neurons(3, 9);

    // Define input, target output, and labels
    vector<double> targets = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    vector<double> label =   {0, 0, 0, 0, 0, 0, 1, 0, 0};
    vector<double> inputs =  {5, 2, 4, 4, 1, 3, 8, 6, 7};

    // Hyperparameters
    double learning_rate = 1e-2;
    double decay_rate = 1e-5;
    vector<double> beta = {0.9, 0.9999}; // Momentum factor for Adam
    int step_size = 1000;                // For step decay learning rate

    // Construct Neural net
    auto start_init = high_resolution_clock::now();
    Neural neural(
        num_neurons, Data(inputs).normalization(SYMMETRIC),
        learning_rate, decay_rate, beta,
        LEAKY_RELU, ITDECAY, ADAM, MSE
    );

    neural.set_step_size(step_size);
    neural.set_target(Data(targets).normalization(SYMMETRIC));
    neural.set_input(Data(inputs).normalization(SYMMETRIC));
    neural.set_hardlabeling(label);
    //neural.set_softlabeling(label, 1e-2);

    int count = 0;
    int process = 0;

    while (count < MAX_COUNT) {
        auto start = high_resolution_clock::now();

        // Perform feedforward and error calculations
        neural.feedforward();
        neural.debug_feedforward();
        neural.debug_weight();
        neural.debug_loss();
        neural.debug_probability();
        neural.debug_error();
        neural.probability_calculation();

        // Safely retrieve the error vector
        vector<double> errors = neural.get_error();
        if (errors.empty()) {
            cerr << "Error: Error vector is empty. Skipping iteration." << endl;
            process++;
            continue; // Skip this iteration to avoid segmentation fault.
        }

        // Count errors below the threshold
        count = count_errors_below_threshold(errors, ERROR_THRESHOLD);

        // Perform backpropagation
        neural.backpropagation();

        // Measure and display the duration of the process
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);

        cout << "Process: " << process
             << " | Time taken: " << duration.count() << " microseconds"
             << " | Errors below threshold: " << count << "/" << MAX_COUNT << endl;

        // Check for convergence
        if (count == MAX_COUNT) {
            cout << "Benchmark test has been successfully completed." << endl;
            break;
        }

        process++;
        count = 0; // Reset count for the next iteration
    }

    return 0;
}