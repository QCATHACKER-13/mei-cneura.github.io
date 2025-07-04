#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

/*double dot_product(const vector<double>& a, const vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += (a[i] * b[i]);
    }
    return sum;
}*/
double dot_product(const vector<double>& a, const vector<double>& b) {
    assert(a.size() == b.size());
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}


vector<double> mat_vec_mult(const vector<vector<double>>& matrix, const vector<double>& vec) {
    std::vector<double> result(matrix.size(), 0.0);
    for (size_t i = 0; i < matrix.size(); ++i) {
        result[i] = dot_product(matrix[i], vec);
    }
    return result;
}