// tests/test_rapids.cpp
#include <iostream>
#include "utils/data_loader.h"
#include "utils/timer.h"
#include "gpu/rapids_wrapper.h"

using namespace semantic_search;
using namespace semantic_search::utils;
using namespace semantic_search::gpu;

int main() {
    std::cout << "RAPIDS cuML test" << std::endl;
    
    // Generate sample data
    size_t numVectors = 10000;
    size_t dimensions = 128;
    
    auto dataset = DataLoader::generateSampleData(numVectors, dimensions);
    
    // Create RAPIDS search
    auto rapidsSearch = std::make_shared<RapidsSearch>();
    
    try {
        rapidsSearch->setDataset(dataset);
        
        // Generate a query vector
        auto queryDataset = DataLoader::generateSampleData(1, dimensions);
        std::vector<float> query(dimensions);
        const float* queryVec = queryDataset->getVector(0);
        std::copy(queryVec, queryVec + dimensions, query.begin());
        
        // Test search
        Timer timer("RAPIDS search");
        auto results = rapidsSearch->search(query, 10);
        
        // Print results
        std::cout << "Results size: " << results.size() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
    
    std::cout << "RAPIDS test completed!" << std::endl;
    return 0;
}