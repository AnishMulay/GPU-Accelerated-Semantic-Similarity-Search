// src/cpu/brute_force.cpp
#include "cpu/brute_force.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <queue>

namespace semantic_search {
namespace cpu {

BruteForceSearch::BruteForceSearch() {
}

BruteForceSearch::~BruteForceSearch() {
}

void BruteForceSearch::setDataset(VectorDataPtr dataset) {
    m_dataset = dataset;
}

std::vector<SearchResult> BruteForceSearch::search(const std::vector<float>& query, int k) {
    if (!m_dataset) {
        throw std::runtime_error("Dataset not set");
    }
    
    if (query.size() != m_dataset->getDimensions()) {
        throw std::invalid_argument("Query vector dimension mismatch");
    }
    
    // Check if k is valid
    size_t numVectors = m_dataset->getSize();
    if (k <= 0 || static_cast<size_t>(k) > numVectors) {
        k = static_cast<int>(numVectors);
    }
    
    // Create a priority queue for top-k results
    std::vector<SearchResult> results;
    results.reserve(numVectors);
    
    // Calculate similarity for each vector in the dataset
    for (size_t i = 0; i < numVectors; i++) {
        const float* vector = m_dataset->getVector(i);
        float similarity = cosineSimilarity(query.data(), vector, m_dataset->getDimensions());
        
        results.emplace_back(static_cast<uint32_t>(i), similarity, m_dataset->getLabel(i));
    }
    
    // Sort results by similarity (descending)
    std::partial_sort(results.begin(), results.begin() + k, results.end());
    
    // Return top-k results
    return std::vector<SearchResult>(results.begin(), results.begin() + k);
}

float BruteForceSearch::cosineSimilarity(const float* a, const float* b, size_t dimensions) const {
    float dotProduct = 0.0f;
    float normA = 0.0f;
    float normB = 0.0f;
    
    for (size_t i = 0; i < dimensions; i++) {
        dotProduct += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    
    // Avoid division by zero
    if (normA == 0.0f || normB == 0.0f) {
        return 0.0f;
    }
    
    // Return cosine similarity
    return dotProduct / (std::sqrt(normA) * std::sqrt(normB));
}

} // namespace cpu
} // namespace semantic_search