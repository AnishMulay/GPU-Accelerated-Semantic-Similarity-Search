// src/utils/vector_data.cpp
#include "utils/vector_data.h"
#include <stdexcept>

namespace semantic_search {

VectorData::VectorData(size_t dimensions) 
    : m_dimensions(dimensions) {
}

void VectorData::addVector(const std::vector<float>& vector, const std::string& label) {
    if (vector.size() != m_dimensions) {
        throw std::invalid_argument("Vector dimension mismatch");
    }
    
    // Add vector elements to the flattened array
    m_vectors.insert(m_vectors.end(), vector.begin(), vector.end());
    
    // Add label
    m_labels.push_back(label);
}

const float* VectorData::getVector(size_t index) const {
    if (index >= getSize()) {
        throw std::out_of_range("Vector index out of range");
    }
    
    // Return pointer to the start of the vector in the flattened array
    return &m_vectors[index * m_dimensions];
}

const std::string& VectorData::getLabel(size_t index) const {
    if (index >= getSize()) {
        throw std::out_of_range("Label index out of range");
    }
    
    return m_labels[index];
}

const float* VectorData::getVectorsData() const {
    if (m_vectors.empty()) {
        return nullptr;
    }
    
    return m_vectors.data();
}

void VectorData::printStats() const {
    std::cout << "Dataset Statistics:" << std::endl;
    std::cout << "  Dimensions: " << m_dimensions << std::endl;
    std::cout << "  Vectors: " << getSize() << std::endl;
    std::cout << "  Total Memory: " << (m_dimensions * getSize() * sizeof(float)) / (1024 * 1024) << " MB" << std::endl;
}

} // namespace semantic_search