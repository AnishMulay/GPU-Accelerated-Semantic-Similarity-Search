// include/utils/data_loader.h
#pragma once
#include <string>
#include <vector>
#include <memory>
#include "vector_data.h"

namespace semantic_search {
namespace utils {

class DataLoader {
public:
    // Function to load vector datasets
    static VectorDataPtr loadGloveVectors(const std::string& filename);
    static VectorDataPtr loadSIFTVectors(const std::string& filename);
    static VectorDataPtr loadCustomEmbeddings(const std::string& filename);
    
    // Helper function to generate sample data for testing
    static VectorDataPtr generateSampleData(size_t numVectors, size_t dimensions);
};

} // namespace utils
} // namespace semantic_search