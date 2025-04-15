// src/cpu/brute_force_omp.cpp
#include "cpu/brute_force_omp.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <queue>
#include <omp.h>

namespace semantic_search {
namespace cpu {

BruteForceSearchOMP::BruteForceSearchOMP() {
}

BruteForceSearchOMP::~BruteForceSearchOMP() {
}

void BruteForceSearchOMP::setDataset(VectorDataPtr dataset) {
    m_dataset = dataset;
}

std::vector<SearchResult> BruteForceSearchOMP::search(const std::vector<float>& query, int k) {
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
    
    // Create vector for results
    std::vector<SearchResult> results;
    results.reserve(numVectors);
    
    // Calculate similarity for each vector in the dataset in parallel
    #pragma omp parallel for
    for (size_t i = 0; i < numVectors; i++) {
        const float* vector = m_dataset->getVector(i);
        float similarity = cosineSimilarity(query.data(), vector, m_dataset->getDimensions());
        
        #pragma omp critical
        {
            results.emplace_back(static_cast<uint32_t>(i), similarity, m_dataset->getLabel(i));
        }
    }
    
    // Sort results by similarity (descending)
    std::partial_sort(results.begin(), results.begin() + k, results.end());
    
    // Return top-k results
    return std::vector<SearchResult>(results.begin(), results.begin() + k);
}

float BruteForceSearchOMP::cosineSimilarity(const float* a, const float* b, size_t dimensions) const {
    float dotProduct = 0.0f;
    float normA = 0.0f;
    float normB = 0.0f;
    
    // Manual loop unrolling for better performance
    size_t i = 0;
    size_t blockSize = 4;
    size_t blocks = dimensions / blockSize;
    size_t remainder = dimensions % blockSize;
    
    // Process in blocks of 4
    for (size_t block = 0; block < blocks; block++, i += blockSize) {
        dotProduct += a[i] * b[i] + a[i+1] * b[i+1] + a[i+2] * b[i+2] + a[i+3] * b[i+3];
        normA += a[i] * a[i] + a[i+1] * a[i+1] + a[i+2] * a[i+2] + a[i+3] * a[i+3];
        normB += b[i] * b[i] + b[i+1] * b[i+1] + b[i+2] * b[i+2] + b[i+3] * b[i+3];
    }
    
    // Process remaining elements
    for (size_t j = 0; j < remainder; j++, i++) {
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