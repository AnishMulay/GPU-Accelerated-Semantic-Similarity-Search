// src/gpu/cuda_search.cpp
#include "gpu/cuda_search.h"
#include "gpu/cuda_kernels.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cuda_runtime.h>

namespace semantic_search {
namespace gpu {

CudaSearch::CudaSearch() 
    : m_deviceVectors(nullptr), m_numVectors(0), m_dimensions(0), m_initialized(false) {
    // Initialize CUDA
    initialize();
}

CudaSearch::~CudaSearch() {
    // Clean up CUDA resources
    cleanup();
}

void CudaSearch::initialize() {
    // Nothing to initialize yet
    m_initialized = true;
}

void CudaSearch::cleanup() {
    // Free device memory
    if (m_deviceVectors != nullptr) {
        freeDeviceMemory(m_deviceVectors);
        m_deviceVectors = nullptr;
    }
}

void CudaSearch::setDataset(VectorDataPtr dataset) {
    if (!m_initialized) {
        throw std::runtime_error("CUDA not initialized");
    }
    
    // Store dataset
    m_dataset = dataset;
    
    // Get dataset properties
    m_numVectors = dataset->getSize();
    m_dimensions = dataset->getDimensions();
    
    // Free old device memory if necessary
    if (m_deviceVectors != nullptr) {
        freeDeviceMemory(m_deviceVectors);
        m_deviceVectors = nullptr;
    }
    
    // Copy vectors to device
    if (m_numVectors > 0) {
        std::cout << "Copying " << m_numVectors << " vectors to GPU..." << std::endl;
        copyVectorsToDevice(dataset->getVectorsData(), &m_deviceVectors, m_numVectors, m_dimensions);
    }
}

std::vector<SearchResult> CudaSearch::search(const std::vector<float>& query, int k) {
    if (!m_initialized) {
        throw std::runtime_error("CUDA not initialized");
    }
    
    if (!m_dataset) {
        throw std::runtime_error("Dataset not set");
    }
    
    if (query.size() != m_dimensions) {
        throw std::invalid_argument("Query vector dimension mismatch");
    }
    
    // Check if k is valid
    if (k <= 0 || static_cast<size_t>(k) > m_numVectors) {
        k = static_cast<int>(m_numVectors);
    }
    
    // Allocate device memory for query
    float* deviceQuery = nullptr;
    copyQueryToDevice(query.data(), &deviceQuery, m_dimensions);
    
    // Allocate device memory for similarities
    float* deviceSimilarities = nullptr;
    size_t similaritiesSize = m_numVectors * sizeof(float);
    cudaMalloc(&deviceSimilarities, similaritiesSize);
    
    // Launch kernel
    launchCosineSimilarityKernel(m_deviceVectors, deviceQuery, deviceSimilarities, 
                              static_cast<int>(m_numVectors), static_cast<int>(m_dimensions));
    
    // Copy similarities back to host
    std::vector<float> similarities(m_numVectors);
    cudaMemcpy(similarities.data(), deviceSimilarities, similaritiesSize, cudaMemcpyDeviceToHost);
    
    // Free device memory
    freeDeviceMemory(deviceQuery);
    freeDeviceMemory(deviceSimilarities);
    
    // Create results vector with indices and similarities
    std::vector<SearchResult> results;
    results.reserve(m_numVectors);
    
    for (size_t i = 0; i < m_numVectors; i++) {
        results.emplace_back(static_cast<uint32_t>(i), similarities[i], m_dataset->getLabel(i));
    }
    
    // Sort results by similarity (descending)
    std::partial_sort(results.begin(), results.begin() + k, results.end());
    
    // Return top-k results
    return std::vector<SearchResult>(results.begin(), results.begin() + k);
}

} // namespace gpu
} // namespace semantic_search