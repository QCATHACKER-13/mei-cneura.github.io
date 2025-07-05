#include <iostream>
#include "cdata_tools.h"

int main() {
    std::vector<double> input = {1, 2, 3, 4, 5};
    std::vector<double> target = {5, 4, 3, 2, 1};

    Data data(input);

    std::cout << "Original input: ";
    for (auto v : input) std::cout << v << " ";
    std::cout << std::endl;

    // Test MINMAX normalization
    auto minmax = data.dataset_normalization(MINMAX);
    std::cout << "MINMAX: ";
    for (auto v : minmax) std::cout << v << " ";
    std::cout << std::endl;

    // Test SYMMETRIC normalization
    Data data2(input);
    auto sym = data2.dataset_normalization(SYMMETRIC);
    std::cout << "SYMMETRIC: ";
    for (auto v : sym) std::cout << v << " ";
    std::cout << std::endl;

    // Test MEANCENTER normalization
    Data data3(input);
    auto meancenter = data3.dataset_normalization(MEANCENTER);
    std::cout << "MEANCENTER: ";
    for (auto v : meancenter) std::cout << v << " ";
    std::cout << std::endl;

    // Test ZSCORE normalization
    Data data4(input);
    auto zscore = data4.dataset_normalization(ZSCORE);
    std::cout << "ZSCORE: ";
    for (auto v : zscore) std::cout << v << " ";
    std::cout << std::endl;

    // Test targetdataset_normalization
    Data data5(input);
    auto tnorm = data5.targetdataset_normalization(MINMAX, target);
    std::cout << "Target MINMAX: ";
    for (auto v : tnorm) std::cout << v << " ";
    std::cout << std::endl;

    // Test targetdata_normalization (single value)
    Data data6(input);
    double single = data6.targetdata_normalization(ZSCORE, 3.0);
    std::cout << "Single value ZSCORE (3.0): " << single << std::endl;

    return 0;
}