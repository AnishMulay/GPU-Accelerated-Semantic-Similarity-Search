// tests/test_faiss.cpp
#include <iostream>
#include "utils/data_loader.h"
#include "utils/timer.h"
#include "cpu/faiss_wrapper.h"

using namespace semantic_search;
using namespace semantic_search::utils;
using namespace semantic_search::cpu;

int main() {
    // Generate sample data
    size_t numVectors = 100000;
    size_t dimensions = 128;
    
    std::cout << "Generating " << numVectors << " vectors with " << dimensions << " dimensions..." << std::endl;
    auto dataset = DataLoader::generateSampleData(numVectors, dimensions);
    
    // Create FAISS index
    int numLists = 100;
    int numCodes = 16;
    int bitsPerCode = 8;
    
    std::cout << "Creating FAISS IVFPQ index with " << numLists << " lists, " 
              << numCodes << " codes, " << bitsPerCode << " bits per code..." << std::endl;
    
    auto faissSearch = std::make_shared<FaissIVFPQ>(numLists, numCodes, bitsPerCode);
    
    // Set dataset and train index
    Timer trainTimer("FAISS index training");
    faissSearch->setDataset(dataset);
    trainTimer.reset();
    
    // Test with different nprobe values
    std::vector<int> nprobeValues = {1, 4, 16, 32, 64};
    
    for (int nprobe : nprobeValues) {
        std::cout << "\nTesting with nprobe = " << nprobe << std::endl;
        faissSearch->setNProbe(nprobe);
        
        // Generate a query vector
        auto queryDataset = DataLoader::generateSampleData(1, dimensions);
        std::vector<float> query(dimensions);
        const float* queryVec = queryDataset->getVector(0);
        std::copy(queryVec, queryVec + dimensions, query.begin());
        
        // Search for nearest neighbors
        Timer searchTimer("FAISS search");
        auto results = faissSearch->search(query, 10);
        
        // Print results
        std::cout << "Top 10 results:" << std::endl;
        for (size_t i = 0; i < results.size(); i++) {
            std::cout << "  " << i + 1 << ": " << results[i].label 
                      << " (similarity: " << results[i].similarity << ")" << std::endl;
        }
    }
    
    std::cout << "FAISS test completed!" << std::endl;
    return 0;
}