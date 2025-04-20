// tests/test_cuda.cpp
#include <iostream>
#include "utils/data_loader.h"
#include "utils/timer.h"
#include "gpu/cuda_search.h"
#include "cpu/brute_force.h"

using namespace semantic_search;
using namespace semantic_search::utils;
using namespace semantic_search::gpu;
using namespace semantic_search::cpu;

int main() {
    // Generate sample data
    std::vector<size_t> dataSizes = {10000, 50000, 100000};
    size_t dimensions = 128;
    
    for (auto size : dataSizes) {
        std::cout << "\n=== Testing with " << size << " vectors ===" << std::endl;
        
        // Generate dataset
        auto dataset = DataLoader::generateSampleData(size, dimensions);
        
        // Create search algorithms
        auto cpuSearch = std::make_shared<BruteForceSearch>();
        cpuSearch->setDataset(dataset);
        
        auto gpuSearch = std::make_shared<CudaSearch>();
        gpuSearch->setDataset(dataset);
        
        // Generate a query vector
        auto queryDataset = DataLoader::generateSampleData(1, dimensions);
        std::vector<float> query(dimensions);
        const float* queryVec = queryDataset->getVector(0);
        std::copy(queryVec, queryVec + dimensions, query.begin());
        
        // Test CPU search
        std::vector<SearchResult> cpuResults;
        {
            Timer timer("CPU brute force");
            cpuResults = cpuSearch->search(query, 10);
        }
        
        // Test GPU search
        std::vector<SearchResult> gpuResults;
        {
            Timer timer("GPU CUDA");
            gpuResults = gpuSearch->search(query, 10);
        }
        
        // Compare results
        std::cout << "\nComparing top 10 results:" << std::endl;
        std::cout << "CPU\t\t\t\tGPU" << std::endl;
        std::cout << "--------------------------------------------" << std::endl;
        
        for (int i = 0; i < 10; i++) {
            std::cout << cpuResults[i].label << " (" << cpuResults[i].similarity << ")\t";
            std::cout << gpuResults[i].label << " (" << gpuResults[i].similarity << ")" << std::endl;
        }
    }
    
    std::cout << "\nCUDA search test completed!" << std::endl;
    return 0;
}