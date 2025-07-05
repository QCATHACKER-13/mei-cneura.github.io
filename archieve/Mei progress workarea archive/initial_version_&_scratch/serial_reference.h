#include <nlohmann/json.hpp>
#include <fstream>
using json = nlohmann::json;

enum class Format { JSON, CSV, TXT };


void save_as_json(const std::string& filename) const {
    json j;
    j["weights"] = weight;
    j["bias"] = bias;
    j["learning_rate"] = learning_rate;
    j["momentum"] = momentum;
    j["timestep"] = timestep;
    j["beta"] = beta;
    j["gradient_weight"] = gradient_weight;
    // Add other parameters as needed

    std::ofstream out(filename);
    out << j.dump(4);
}

void save_as_csv(const std::string& filename) const {
    std::ofstream out(filename);
    for (size_t i = 0; i < weight.size(); ++i) {
        out << weight[i];
        if (i != weight.size() - 1) out << ",";
    }
    out << "\n" << bias << "\n"; // Optional: Bias on second line
}

void save_as_txt(const std::string& filename) const {
    std::ofstream out(filename);
    out << "Weights:\n";
    for (const auto& w : weight) out << w << " ";
    out << "\nBias: " << bias;
    out << "\nLearning Rate: " << learning_rate;
    out << "\nTimestep: " << timestep;
    out << "\nGradient Weight:\n";
    for (const auto& g : gradient_weight) out << g << " ";
    // Extend with more info as needed
}


void load_from_json(const std::string& filename) {
    json j;
    std::ifstream in(filename);
    in >> j;

    weight = j["weights"].get<std::vector<double>>();
    bias = j["bias"].get<double>();
    learning_rate = j["learning_rate"].get<double>();
    momentum = j["momentum"].get<std::vector<std::vector<double>>>();
    timestep = j["timestep"].get<int>();
    beta = j["beta"].get<std::vector<double>>();
    gradient_weight = j["gradient_weight"].get<std::vector<double>>();
    // Load others as needed
}

void load_from_csv(const std::string& filename) {
    std::ifstream in(filename);
    std::string line;
    
    if (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string val;
        weight.clear();
        while (std::getline(ss, val, ',')) {
            weight.push_back(std::stod(val));
        }
    }

    if (std::getline(in, line)) {
        bias = std::stod(line); // second line for bias
    }
}

void save(const std::string& filename, Format format) const {
    switch (format) {
        case Format::JSON: save_as_json(filename); break;
        case Format::CSV:  save_as_csv(filename);  break;
        case Format::TXT:  save_as_txt(filename);  break;
    }
}

void load(const std::string& filename, Format format) {
    switch (format) {
        case Format::JSON: load_from_json(filename); break;
        case Format::CSV:  load_from_csv(filename);  break;
        case Format::TXT:  /* Not implemented */     break;
    }
}


