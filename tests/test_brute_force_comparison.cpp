// tests/test_brute_force_comparison.cpp
#include <iostream>
#include "utils/data_loader.h"
#include "utils/timer.h"
#include "cpu/brute_force.h"
#include "cpu/brute_force_omp.h"
#include <omp.h>

using namespace semantic_search;
using namespace semantic_search::utils;
using namespace semantic_search::cpu;

int main() {
    // Print number of available threads
    std::cout << "OpenMP maximum threads: " << omp_get_max_threads() << std::endl;
    
    // Generate sample data of increasing sizes
    std::vector<size_t> dataSizes = {10000, 50000, 100000};
    std::vector<size_t> dimensions = {64, 128, 256};
    
    for (auto dim : dimensions) {
        std::cout << "\n=== Testing with " << dim << " dimensions ===" << std::endl;
        
        for (auto size : dataSizes) {
            std::cout << "\nDataset size: " << size << std::endl;
            
            // Generate dataset
            auto dataset = DataLoader::generateSampleData(size, dim);
            
            // Create search algorithms
            auto bruteForce = std::make_shared<BruteForceSearch>();
            bruteForce->setDataset(dataset);
            
            auto bruteForceOMP = std::make_shared<BruteForceSearchOMP>();
            bruteForceOMP->setDataset(dataset);
            
            // Generate a query vector
            auto queryDataset = DataLoader::generateSampleData(1, dim);
            std::vector<float> query(dim);
            const float* queryVec = queryDataset->getVector(0);
            std::copy(queryVec, queryVec + dim, query.begin());
            
            // Test regular brute force
            {
                Timer timer("Regular brute force");
                auto results = bruteForce->search(query, 10);
                std::cout << "  Top result similarity: " << results[0].similarity << std::endl;
            }
            
            // Test OpenMP brute force
            {
                Timer timer("OpenMP brute force");
                auto results = bruteForceOMP->search(query, 10);
                std::cout << "  Top result similarity: " << results[0].similarity << std::endl;
            }
        }
    }
    
    std::cout << "\nBrute force comparison completed!" << std::endl;
    return 0;
}