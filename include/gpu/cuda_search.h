// include/gpu/cuda_search.h
#pragma once
#include "similarity_search.h"
#include <vector>
#include <string>
#include <cuda_runtime.h>

namespace semantic_search {
namespace gpu {

class CudaSearch : public SimilaritySearch {
public:
    CudaSearch();
    ~CudaSearch() override;
    
    // Initialize with a dataset
    void setDataset(VectorDataPtr dataset) override;
    
    // Search for nearest neighbors
    std::vector<SearchResult> search(const std::vector<float>& query, int k) override;
    
    // Get the name of the algorithm
    std::string getName() const override { return "CUDA Cosine Similarity"; }
    
private:
    VectorDataPtr m_dataset;
    float* m_deviceVectors;
    size_t m_numVectors;
    size_t m_dimensions;
    bool m_initialized;
    
    // Initialize CUDA
    void initialize();
    
    // Clean up CUDA resources
    void cleanup();
};

} // namespace gpu
} // namespace semantic_search