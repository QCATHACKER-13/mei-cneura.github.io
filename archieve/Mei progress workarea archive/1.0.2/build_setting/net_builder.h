#include "../1.0.2/cneura.h"
#include "../config_setting/neuralnet_config.h"

class NetBuilder {
public:
    // Build a Neural Network from the given configuration and input data
    static Neural build(const NeuralConfig& config, const std::vector<double>& input) {
        std::vector<size_t> sizes;
        for (const auto& layer : config.layers)
            sizes.push_back(layer.num_neurons);

        Neural net(sizes, input, config.learning_rate, config.decay_rate, config.beta,
                   LEAKY_RELU, ITDECAY, config.optimizer, config.loss); // Use actual map logic if needed

        std::vector<double> keep_prob;
        for (const auto& layer : config.layers)
            keep_prob.push_back(layer.dropout_keep_prob);

        net.set_step_size(config.step_size);
        net.set_dropout(keep_prob);

        return net;
    }
};
