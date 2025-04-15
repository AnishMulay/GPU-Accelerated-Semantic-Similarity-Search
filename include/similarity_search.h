// include/similarity_search.h
#pragma once
#include "utils/vector_data.h"
#include <vector>
#include <cstdint>
#include <string>

namespace semantic_search {

struct SearchResult {
    uint32_t index;
    float similarity;
    std::string label;
    
    // Constructor
    SearchResult(uint32_t idx, float sim, const std::string& lbl)
        : index(idx), similarity(sim), label(lbl) {}
    
    // Comparison operator for sorting
    bool operator<(const SearchResult& other) const {
        return similarity > other.similarity; // For descending order
    }
};

// Base class for all similarity search implementations
class SimilaritySearch {
public:
    virtual ~SimilaritySearch() = default;
    
    // Initialize with a dataset
    virtual void setDataset(VectorDataPtr dataset) = 0;
    
    // Search for nearest neighbors
    virtual std::vector<SearchResult> search(const std::vector<float>& query, int k) = 0;
    
    // Get the name of the algorithm
    virtual std::string getName() const = 0;
};

// Smart pointer type for similarity search algorithms
using SimilaritySearchPtr = std::shared_ptr<SimilaritySearch>;

} // namespace semantic_search