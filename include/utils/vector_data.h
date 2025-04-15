// include/vector_data.h
#pragma once
#include <vector>
#include <string>
#include <memory>
#include <iostream>

namespace semantic_search {

class VectorData {
public:
    VectorData(size_t dimensions);
    
    // Add a vector to the dataset
    void addVector(const std::vector<float>& vector, const std::string& label = "");
    
    // Get dimensions
    size_t getDimensions() const { return m_dimensions; }
    
    // Get number of vectors
    size_t getSize() const { return m_vectors.size(); }
    
    // Access a specific vector
    const float* getVector(size_t index) const;
    
    // Access a specific label
    const std::string& getLabel(size_t index) const;
    
    // Get all vectors as a contiguous array (for efficient processing)
    const float* getVectorsData() const;
    
    // Get all labels
    const std::vector<std::string>& getLabels() const { return m_labels; }
    
    // Print dataset statistics
    void printStats() const;
    
private:
    size_t m_dimensions;
    std::vector<float> m_vectors;  // Flattened for better memory layout
    std::vector<std::string> m_labels;
};

// Smart pointer type for vector data
using VectorDataPtr = std::shared_ptr<VectorData>;

} // namespace semantic_search