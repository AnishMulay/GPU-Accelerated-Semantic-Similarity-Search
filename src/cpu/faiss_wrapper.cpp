// src/cpu/faiss_wrapper.cpp
#include "cpu/faiss_wrapper.h"
#include <stdexcept>
#include <iostream>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>

namespace semantic_search {
namespace cpu {

FaissIVFPQ::FaissIVFPQ(int numLists, int numCodes, int bitsPerCode) 
    : m_numLists(numLists), m_numCodes(numCodes), m_bitsPerCode(bitsPerCode), 
      m_nprobe(1), m_trained(false), m_index(nullptr){
}

FaissIVFPQ::~FaissIVFPQ() {
}

void FaissIVFPQ::setDataset(VectorDataPtr dataset) {
    // Store dataset for label lookup
    m_dataset = dataset;
    
    // Create a new FAISS index
    size_t dimensions = dataset->getDimensions();
    
    m_index.reset();

    // Create a flat index for the coarse quantizer
    faiss::IndexFlatL2* coarseQuantizer = new faiss::IndexFlatL2(dimensions);
    
    // Important: Use proper constructor parameters matching FAISS documentation
    m_index.reset(new faiss::IndexIVFPQ(coarseQuantizer, dimensions, m_numLists, m_numCodes, m_bitsPerCode));
    
    // Take ownership of the quantizer
    m_index->own_fields = true;

    // Train and add vectors if dataset has data
    if (dataset->getSize() > 0) {
        trainIndex();
    }
}

void FaissIVFPQ::trainIndex() {
    if (!m_dataset || m_dataset->getSize() == 0) {
        throw std::runtime_error("Dataset is empty");
    }
    
    size_t numVectors = m_dataset->getSize();
    size_t dimensions = m_dataset->getDimensions();
    
    std::cout << "Training FAISS IVFPQ index with " << numVectors << " vectors..." << std::endl;
    
    // Train the index
    m_index->train(numVectors, m_dataset->getVectorsData());
    
    // Add vectors
    m_index->add(numVectors, m_dataset->getVectorsData());
    
    // Set default nprobe
    m_index->nprobe = m_nprobe;
    
    std::cout << "FAISS index trained and populated with " << numVectors << " vectors" << std::endl;
    
    m_trained = true;
}

void FaissIVFPQ::setNProbe(int nprobe) {
    m_nprobe = nprobe;
    if (m_index) {
        m_index->nprobe = nprobe;
    }
}

std::vector<SearchResult> FaissIVFPQ::search(const std::vector<float>& query, int k) {
    if (!m_index || !m_trained) {
        throw std::runtime_error("Index not trained");
    }
    
    if (query.size() != m_dataset->getDimensions()) {
        throw std::invalid_argument("Query vector dimension mismatch");
    }
    
    // Prepare results arrays
    std::vector<float> distances(k);
    std::vector<faiss::Index::idx_t> indices(k);
    
    try {
        // Set nprobe before search
        m_index->nprobe = m_nprobe;
        
        // Perform search with proper error handling
        m_index->search(1, query.data(), k, distances.data(), indices.data());
    } catch (const std::exception& e) {
        std::cerr << "FAISS search error: " << e.what() << std::endl;
        throw;
    }
    
    // Convert to search results
    std::vector<SearchResult> results;
    results.reserve(k);
    
    for (int i = 0; i < k; i++) {
        // Check if the index is valid (-1 indicates no more results)
        if (indices[i] < 0) {
            break;
        }
        
        // Convert distance to similarity (FAISS uses L2 distance by default)
        float similarity = 1.0f / (1.0f + distances[i]);
        
        // Add to results
        uint32_t idx = static_cast<uint32_t>(indices[i]);
        results.emplace_back(idx, similarity, m_dataset->getLabel(idx));
    }
    
    return results;
}

} // namespace cpu
} // namespace semantic_search