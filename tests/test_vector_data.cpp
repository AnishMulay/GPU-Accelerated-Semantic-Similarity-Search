// tests/test_vector_data.cpp
#include <iostream>
#include "utils/vector_data.h"
#include "utils/data_loader.h"

using namespace semantic_search;
using namespace semantic_search::utils;

int main() {
    // Test generating sample data
    auto sampleData = DataLoader::generateSampleData(1000, 128);
    
    // Print statistics
    sampleData->printStats();
    
    // Test accessing vectors
    std::cout << "First vector data: [";
    const float* firstVector = sampleData->getVector(0);
    for (size_t i = 0; i < 5; i++) {
        std::cout << firstVector[i];
        if (i < 4) std::cout << ", ";
    }
    std::cout << ", ...]" << std::endl;
    
    // Test accessing labels
    std::cout << "First vector label: " << sampleData->getLabel(0) << std::endl;
    
    std::cout << "Vector data tests passed!" << std::endl;
    return 0;
}