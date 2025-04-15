// include/cpu/brute_force_omp.h
#pragma once
#include "similarity_search.h"
#include <vector>
#include <string>

namespace semantic_search {
namespace cpu {

class BruteForceSearchOMP : public SimilaritySearch {
public:
    BruteForceSearchOMP();
    ~BruteForceSearchOMP() override;
    
    // Initialize with a dataset
    void setDataset(VectorDataPtr dataset) override;
    
    // Search for nearest neighbors
    std::vector<SearchResult> search(const std::vector<float>& query, int k) override;
    
    // Get the name of the algorithm
    std::string getName() const override { return "CPU Brute Force (OpenMP)"; }
    
private:
    VectorDataPtr m_dataset;
    
    // Calculate cosine similarity between two vectors
    float cosineSimilarity(const float* a, const float* b, size_t dimensions) const;
};

} // namespace cpu
} // namespace semantic_search