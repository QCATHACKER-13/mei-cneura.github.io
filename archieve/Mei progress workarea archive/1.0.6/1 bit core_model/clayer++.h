/* Project: NEURA 
NEURA is an artificial neuron and network development research project 
focused on artificial intelligence for data analysis. The goal is to find 
the optimal neural network structure and improve artificial neurons for 
efficiency and speed.

Future projects:
- Project Mei: Neuron investigation project.

Developer: Christopher Emmanuelle J. Visperas, Applied Physics Researcher

Note in this neuron testing in this following 
    - beta or momentum factor is set to 0.9 and 0.9999 as default, and must be the constant parameter
    - if beta is not a constant parameter then the target data with in the 
      range of the input data cause sometimes are error and more moderate error if the
      target data is out of range.


*/


#ifndef CLAYER++_H
#define CLAYER++_H
constexpr double ALPHA = 0.01; // Adjust this value
constexpr double EPSILON = 1e-8;
constexpr double CLIP_GRAD = 10.0; // Gradient clipping limit

#include <iostream>
//#include <cstdlib>   // for rand() and srand()
//#include <ctime>     // for seeding rand()
#include <cmath>     // for exp()
#include <iomanip>   // for extension debugging
#include <random>    // For random number generation
#include <vector>
#include <algorithm> // for max_element, accumulate
#include <numeric>   // for accumulate
#include <cassert>   // for assert
//#include <memory>

using namespace std;

// Enum for layer types
enum neurontype { INPUT, HIDDEN, OUTPUT }; // Enum for neuron types
enum normtype { MINMAX, SYMMETRIC, MEANCENTER, ZSCORE }; // Enum for normalization types
enum actfunc { SIGMOID, RELU, TANH, LEAKY_RELU, ELU}; // Enum for activation functions
enum optimizer { SGD, ADAGRAD, RMSPROP, ADAM}; // Enum for optimization algorithms
enum lrs { CONSTANT, STEPDECAY, EXPDECAY, ITDECAY}; // Enum for learning rate adjustment strategies
enum lossfunc { MSE, BCE, CCE, HUBER}; // Enum for loss functions
enum lrmode {SUPERVISED, UNSUPERVISED, REINFORCEMENT, SELF_LEARNING, MULTI_TASK}; // Enum for learning modes

// The Neural Layer class represents a single artificial neuron with support 
// for various activation functions, optimizers, and learning rate schedules.
// It includes features such as batch normalization, momentum-based optimization,
// and adaptive learning rate updates.

class Layer {
    private:
        //neuron's parameter
        int step_size, timestep = 0; // Training step parameters

        //neuron's property
        vector<double> input,
            output, activated_output,
            bias, gradient_bias,
            beta, //momentum factor for Adam optimizer
            m_bias,//momentum in bias
            ; // Regularization error for weights
        
        //neuron's weight and bias
        vector <vector<double>> weight, gradient_weight, error_Lw;
        
        vector<vector<double>> momentum; // Momentum storage for optimizers
        double output, target, bias, 
            error, error_Lb, activated_output, 
            gradient_bias, 
            learning_rate, decay_rate; 
        
        vector<double> error_history; // Error history for analysis
        actfunc actFunc; // Activation function type
        lrs lr_schedule; // Learning rate adjustment strategy
        optimizer opt; // Optimization algorithm
        lossfunc lossFunc; // Loss function type
        neurontype ntype; // Neuron type (input, hidden, output)
        normtype ntype_norm; // Normalization type

        double randomInRange(double min, double max) {
            assert(min < max); // Ensure valid range
            static thread_local random_device rd; // Non-deterministic random device (if available)
            static thread_local mt19937 generator(rd()); // Mersenne Twister RNG seeded by random device
            uniform_real_distribution<double> distribution(min, max); // Uniform distribution between min and max
            return distribution(generator);
        }

        // Activation functions and their derivatives
        
        double activation_value(double x) {
            switch (actFunc) {
                case SIGMOID: return 1.0 / (1.0 + exp(-x));
                case RELU: return max(0.0, x);
                case TANH: return tanh(x);
                case LEAKY_RELU: return x > 0 ? x : ALPHA * x;
                case ELU: return x >= 0 ? x : ALPHA * (exp(x) - 1);
                default: return x;
            }
        }
        
        double activation_derivative(double x) {
            switch (actFunc) {
                case SIGMOID: {
                    double s = activation_value(x);
                    return s * (1.0 - s);
                }
                case RELU: return x > 0 ? 1.0 : 0.0;
                case TANH: {
                    double t = tanh(x);
                    return 1.0 - t * t;
                }
                case LEAKY_RELU: return x > 0 ? 1.0 : ALPHA;
                case ELU: return x >= 0 ? 1.0 : ALPHA * std::exp(x);
                default: return 1.0;
            }
        }

        // Updates the learning rate based on the selected scheduling method
        void update_learning_rate() {
            switch (lr_schedule) {
                case CONSTANT:
                    // Keep learning rate unchanged
                    //initial_error = error;
                    break;
                    
                case STEPDECAY:
                    if (step_size > 0 && timestep % step_size == 0) {
                        //learning_rate[1] = learning_rate[1] * decay_rate; // Prevent underflow
                        learning_rate *= decay_rate;
                    }
                    break;
                
                case EXPDECAY:
                    //learning_rate[1] = learning_rate[1] * exp(-decay_rate * timestep);
                    learning_rate *= exp(-decay_rate * timestep);
                    break;
                    
                case ITDECAY:
                    //learning_rate[1] = learning_rate[1] / (1 + (decay_rate * timestep));
                    learning_rate /= (1 + (decay_rate * timestep));
                    break;
            }
        }

    public:
        // Constructor for initializing neuron properties
        Neuron(const vector<double>& inputs, 
            const double& learning_rate, double decay_rate, const vector<double>& beta,
            neurontype ntype, actfunc actFunc, lrs lr_schedule, optimizer opt, lossfunc lossFunc)
            
            : input(inputs),
            learning_rate(learning_rate), decay_rate(decay_rate), beta(beta),
            ntype(ntype), actFunc(actFunc), lr_schedule(lr_schedule), opt(opt), lossFunc(lossFunc)
            
            {
                // Initialize weight and bias vectors
                assert(!inputs.empty()); // Ensure inputs are non-empty
                
                momentum.assign(2, vector<double>(inputs.size(), 0.0));
                m_bias.assign(2, 0.0);
        }

        void initialize() {
            weight.assign(input.size(), 0.0);
            gradient_weight.assign(input.size(), 0.0);
            error_Lw.resize(weight.size(), 0.0);

            // Ensure weights are initialized
            /*assert(weight.size() == input.size() &&
                weight.size() == gradient_weight.size() &&
                input.size() == gradient_weight.size()
            );*/

            double scale = 1.0;

            //Choose appropriate scaling factor based on activation function
            if (actFunc == RELU || actFunc == LEAKY_RELU) {
                scale = sqrt(2.0 / input.size());  // He Initialization
            } else if (actFunc == SIGMOID || actFunc == TANH) {
                scale = sqrt(1.0 / input.size());  // Xavier Initialization
            }

            // Initialize weights with the chosen scaling
            for (size_t i = 0; i < weight.size(); ++i) {
                assert(i < weight.size()); // Ensure index is within limits
                weight[i] = randomInRange(-scale, scale);
            }
        
            // Initialize bias separately
            if (actFunc == RELU || actFunc == LEAKY_RELU) {
                bias = 0.01;  // Small positive bias to prevent dead neurons
            } else {
                bias = randomInRange(-scale, scale);  // Xavier-based random bias for other activations
            }
        }

        void normalization(normtype ntype_norm) {
            this -> ntype_norm = ntype_norm;
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
                    for(size_t i = 0; i < input.size(); i++){
                        assert(i < input.size()); // Ensure index is within limits
                        input[i] = ((input[i] - min)/range);
                        if(ntype == OUTPUT){target = (target - min)/range;}
                    }
                    break;

                case SYMMETRIC:
                    // Implement symmetric normalization
                    for (size_t i = 0; i < input.size(); i++){
                        assert(i < input.size()); // Ensure index is within limits
                        input[i] = (2 * ((input[i] - min) / range)) - 1;
                    }
                    if(ntype == OUTPUT){target = (2 * ((target - min) / range)) - 1;}
                    break;

                case MEANCENTER:
                    // Implement mean centering
                    for (size_t i = 0; i < input.size(); i++){
                        assert(i < input.size()); // Ensure index is within limits
                        input[i] = (2 * ((input[i] - mean) / range)) - 1;
                    }
                    if(ntype == OUTPUT){target = (2 * ((target - mean) / range)) - 1;}
                    break;

                case ZSCORE:
                    // Implement z-score normalization
                    for (size_t i = 0; i < input.size(); i++){
                        assert(i < input.size()); // Ensure index is within limits
                        input[i] = (input[i] - mean) / stddev;
                    }
                    if(ntype == OUTPUT){target = (target - mean) / stddev;}
                    break;
            }
        }

        // Computes the neuron's output using feedforward propagation
        void feedforward() {
            this->output = inner_product(input.begin(), input.end(), weight.begin(), bias);

            assert(!isnan(output)); // Check for NaN output
            
            this->activated_output = activation_value(output);
        }

        void regularizated(){
            for (size_t i = 0; i < weight.size(); ++i) {
                assert(i < weight.size()); // Ensure index is within limits
                
                if(weight[i] > 0){error_Lw[i] = (2 * weight[i]) + 1;}
                else if(weight[i] < 0){error_Lw[i] = (-2 * weight[i]) - 1;}
            }

            if(bias > 0){error_Lb =  (2 * bias) + 1;}
            else if(bias < 0){error_Lb =  (-2 * bias) - 1;}
        }

        //To calculate the error between output and target
        void loss_derivative() {
            switch (lossFunc) {
                case MSE:
                    error = target - activated_output; // Mean squared error
                    break;
                    
                case BCE:
                    error = -(target / activated_output) + ((1 - target) / (1 - activated_output)); //Binary cross entropy
                    break;
                    
                case CCE:
                    error = -(target / activated_output); //Categorical cross entropy
                    break;
                    
                case HUBER:
                    double delta = 1; // Huber loss delta
                    if (abs(target - activated_output) <= delta) {
                        error =  target - activated_output;  // MSE region
                        break;
                    } else {
                        error = delta * (target - activated_output > 1 ? 1 : -1);  // MAE region
                        break;
                    }
            }
        }
        
        // Performs backpropagation to update weights and bias
        void backward() {
            assert(timestep >= 0); // Ensure valid timestep
            timestep++;
            regularizated(); // Regularization
            update_learning_rate();  // Adjust learning rate dynamically
            update_weights(); // Update weights
            update_bias(); // Update bias
        }

        void update_weights(){
            #pragma omp parallel for 
            //Updating the weights
            for (size_t i = 0; i < weight.size(); ++i) {
                assert(i < weight.size()); // Ensure index is within limits
                gradient_weight[i] = (-(error + error_Lw[i]) * activation_derivative(output) * input[i]);

                switch (opt) {
                    case SGD:
                        this -> momentum[0][i] = ((beta[0] * momentum[0][i]) + ((1 - beta[0]) * gradient_weight[i]));
                        this -> weight[i] -= (learning_rate * momentum[0][i]);
                        break;
                    
                    case ADAGRAD:
                        this -> momentum[0][i] += pow(gradient_weight[i], 2);
                        this -> weight[i] -= ((learning_rate * gradient_weight[i]) / (sqrt(momentum[0][i]) + EPSILON));
                        break;
                        
                    case RMSPROP:
                        this -> momentum[0][i] = ((beta[0] * momentum[0][i]) + ((1 - beta[0]) * pow(gradient_weight[i], 2)));
                        this -> weight[i] -= ((learning_rate * gradient_weight[i]) / (sqrt(momentum[0][i] + EPSILON)));
                        break;
                        
                    case ADAM:
                        // Compute first and second moment estimates                   

                        this -> momentum[0][i] = ((beta[0] * momentum[0][i]) + ((1 - beta[0]) * gradient_weight[i]));
                        this -> momentum[1][i] = ((beta[1] * momentum[1][i]) + ((1 - beta[1]) * pow(gradient_weight[i], 2)));

                        // Correct bias
                        double m_hat = (momentum[0][i] / (1 - pow(beta[0], timestep)));
                        double v_hat = (momentum[1][i] / (1 - pow(beta[1], timestep)));
                        
                        // Apply Adam update rule
                        this -> weight[i] -= ((learning_rate * m_hat) / (sqrt(v_hat) + EPSILON));
                        break;
                    }
            }
        }

        void update_bias(){
            // Bias update (same logic as weight updates)
            gradient_bias = (-(error + error_Lb) * activation_derivative(output));
            switch (opt) {
                case SGD:
                    this -> bias -= (learning_rate * gradient_bias);
                    break;
                
                case ADAGRAD:
                    this -> m_bias[0] += pow(gradient_bias, 2);
                    this -> bias -= ((learning_rate * gradient_bias) / (sqrt(m_bias[0]) + EPSILON));
                    break;
                    
                case RMSPROP:
                    this -> m_bias[0] = ((beta[0] * m_bias[0]) + ((1 - beta[0]) * pow(gradient_bias, 2)));
                    this -> bias -= ((learning_rate * gradient_bias) / (sqrt(m_bias[0]) + EPSILON));
                    break;
                    
                case ADAM:
                    this -> m_bias[0] = ((beta[0] * m_bias[0]) + ((1 - beta[0]) * gradient_bias));
                    this -> m_bias[1] = ((beta[1] * m_bias[1]) + ((1 - beta[1]) * pow(gradient_bias, 2)));
                    
                    double beta1_correction = max(1 - pow(beta[0], timestep), EPSILON);
                    double beta2_correction = max(1 - pow(beta[1], timestep), EPSILON);
                    double m_bias_hat = (m_bias[0] / beta1_correction);
                    double v_bias_hat = (m_bias[1] / beta2_correction);
                    
                    this -> bias -= ((learning_rate * m_bias_hat) / (sqrt(max(v_bias_hat, EPSILON)) + EPSILON));
                    break;
            }
        }

        // Prints the neuron's current state (useful for debugging)
        void print_neuron(size_t id) {
            int col1_width = 10, col2_width = 15, col3_width = 20;

            // Print table row
            cout << id << setw(col1_width) << activated_output 
            << setw(col2_width) << bias;
            for (size_t i = 0; i < weight.size(); ++i) {
                assert(i < weight.size()); // Ensure index is within limits
                cout << setw(col3_width) << weight[i];
            }
            //cout << setw(col3_width) << prediction;
            cout << setw(col3_width) << error;
            cout << setw(col3_width) << learning_rate;
            cout << endl;
        }

        // Trains the neuron for a given number of epochs until the error is below a threshold
        void training(size_t id, int num_epochs, double error_margin, bool switcher){
            assert(error_margin > 0); // Validate error margin
            error_history.reserve(num_epochs); // Preallocate memory

            // Print the normalize input
            cout << "Input: ";
            for (size_t i = 0; i < input.size(); ++i) {
                assert(i < input.size()); // Ensure index is within limits
                cout << setw(10) << input[i];
            }
            cout << setw(20) << target << endl << endl;

            for (int epoch = 1; epoch <= num_epochs; ++epoch) {
                //train_epoch(id, error_margin, switcher);
                assert(error_margin > 0); // Validate error margi
                feedforward();
                if (switcher) {print_neuron(id);}
                if (abs(error) < error_margin) {break;}// Stop training if error margin is reached
                regularizated();
                backward();
            }
        }
        
        /*void train_epoch(size_t id, double error_margin, bool switcher) {
        }*/

        // Getter functions for neuron parameters
        vector <double> get_input() {return input;}
        double get_output() {return output;}
        double get_activated_output() {return activated_output;}
        double get_bias() {return bias;}
        double get_gradient_bias() {return gradient_bias;}
        vector<double> get_gradient_weight() {return gradient_weight;}
        vector<double> get_weight() {return weight;}
        double get_error() {return error;}
        double get_learning_rate() {return learning_rate;}
        double get_timestep(){return timestep;}

        // Setter functions for neuron parameter
        void set_step_size(const int& stepsize){this -> step_size = stepsize;}
        void set_target(const double& targets){this -> target = targets;} //Setting the target of the neuron
        void set_weight(const vector<double>& weights) {
            assert(weights.size() == weight.size() && weights.size() == input.size()); // Ensure sizes match
            this->weight = weights;
        }
        void set_bias(double biases) {this -> bias = biases;} //Setting the bias of the neuron
        void set_error(double errors) {this -> error = errors;} //Setting the error of the neuron
        void set_learning_rate(const double& learning_rate) {this -> learning_rate = learning_rate;} //Setting the learning rate
};

#endif