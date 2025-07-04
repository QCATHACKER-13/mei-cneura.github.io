#include <iostream>
#include <iomanip>  // For std::setw

int main() {
    // Define column widths
    int col1_width = 10, col2_width = 15, col3_width = 20;

    // Print table header
    std::cout << std::left << std::setw(col1_width) << "ID"
              << std::setw(col2_width) << "Name"
              << std::setw(col3_width) << "Value" 
              << std::endl;
    std::cout << std::string(45, '-') << std::endl; // Print separator

    // Print table rows
    std::cout << std::setw(col1_width) << "1"
              << std::setw(col2_width) << "Sensor_A"
              << std::setw(col3_width) << "25.6"
              << std::endl;

    std::cout << std::setw(col1_width) << "2"
              << std::setw(col2_width) << "Sensor_B"
              << std::setw(col3_width) << "30.2"
              << std::endl;

    std::cout << std::setw(col1_width) << "3"
              << std::setw(col2_width) << "Sensor_C"
              << std::setw(col3_width) << "28.9"
              << std::endl;

    return 0;
}
