// src/utils/data_loader.cpp
#include "utils/data_loader.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>

namespace semantic_search {
namespace utils {

VectorDataPtr DataLoader::loadGloveVectors(const std::string& filename) {
    std::cout << "Loading GloVe vectors from: " << filename << std::endl;
    
    // Create a new VectorData object with 300 dimensions (GloVe)
    auto vectorData = std::make_shared<VectorData>(300);
    
    // Placeholder for actual file loading
    std::cout << "GloVe loading not yet implemented - returning empty dataset" << std::endl;
    
    return vectorData;
}

VectorDataPtr DataLoader::loadSIFTVectors(const std::string& filename) {
    std::cout << "Loading SIFT vectors from: " << filename << std::endl;
    
    // Create a new VectorData object with 128 dimensions (SIFT)
    auto vectorData = std::make_shared<VectorData>(128);
    
    // Placeholder for actual file loading
    std::cout << "SIFT loading not yet implemented - returning empty dataset" << std::endl;
    
    return vectorData;
}

VectorDataPtr DataLoader::loadCustomEmbeddings(const std::string& filename) {
    std::cout << "Loading custom embeddings from: " << filename << std::endl;
    
    // Create a new VectorData object with 768 dimensions (BERT)
    auto vectorData = std::make_shared<VectorData>(768);
    
    // Placeholder for actual file loading
    std::cout << "Custom embeddings loading not yet implemented - returning empty dataset" << std::endl;
    
    return vectorData;
}

VectorDataPtr DataLoader::generateSampleData(size_t numVectors, size_t dimensions) {
    std::cout << "Generating " << numVectors << " sample vectors with " << dimensions << " dimensions" << std::endl;
    
    auto vectorData = std::make_shared<VectorData>(dimensions);
    
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    // Generate random vectors
    for (size_t i = 0; i < numVectors; i++) {
        std::vector<float> vector(dimensions);
        float norm = 0.0f;
        
        // Generate random values
        for (size_t j = 0; j < dimensions; j++) {
            vector[j] = dist(gen);
            norm += vector[j] * vector[j];
        }
        
        // Normalize to unit length (for cosine similarity)
        norm = std::sqrt(norm);
        for (size_t j = 0; j < dimensions; j++) {
            vector[j] /= norm;
        }
        
        // Add to dataset
        vectorData->addVector(vector, "sample_" + std::to_string(i));
    }
    
    return vectorData;
}

} // namespace utils
} // namespace semantic_search