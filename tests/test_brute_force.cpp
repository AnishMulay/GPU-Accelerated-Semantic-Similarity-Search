// tests/test_brute_force.cpp
#include <iostream>
#include "utils/data_loader.h"
#include "utils/timer.h"
#include "cpu/brute_force.h"

using namespace semantic_search;
using namespace semantic_search::utils;
using namespace semantic_search::cpu;

int main() {
    // Generate sample data
    auto dataset = DataLoader::generateSampleData(10000, 128);
    dataset->printStats();
    
    // Create brute force search
    auto search = std::make_shared<BruteForceSearch>();
    search->setDataset(dataset);
    
    // Generate a query vector
    auto queryDataset = DataLoader::generateSampleData(1, 128);
    std::vector<float> query(128);
    const float* queryVec = queryDataset->getVector(0);
    std::copy(queryVec, queryVec + 128, query.begin());
    
    // Search for nearest neighbors
    std::cout << "Searching for nearest neighbors..." << std::endl;
    Timer timer("Brute force search");
    auto results = search->search(query, 10);
    
    // Print results
    std::cout << "Top 10 results:" << std::endl;
    for (size_t i = 0; i < results.size(); i++) {
        std::cout << "  " << i + 1 << ": " << results[i].label 
                  << " (similarity: " << results[i].similarity << ")" << std::endl;
    }
    
    std::cout << "Brute force search test completed!" << std::endl;
    return 0;
}