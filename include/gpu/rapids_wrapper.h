// include/gpu/rapids_wrapper.h
#pragma once
#include "similarity_search.h"
#include <vector>
#include <string>
#include <memory>

// Forward declarations to avoid including RAPIDS headers
namespace ML {
    namespace Metrics {
        class DistanceType;
    }
}

namespace semantic_search {
namespace gpu {

class RapidsSearch : public SimilaritySearch {
public:
    RapidsSearch();
    ~RapidsSearch() override;
    
    // Initialize with a dataset
    void setDataset(VectorDataPtr dataset) override;
    
    // Search for nearest neighbors
    std::vector<SearchResult> search(const std::vector<float>& query, int k) override;
    
    // Get the name of the algorithm
    std::string getName() const override { return "RAPIDS cuML"; }
    
private:
    VectorDataPtr m_dataset;
    
    // Device pointers
    void* m_deviceVectors;
    size_t m_numVectors;
    size_t m_dimensions;
    bool m_initialized;
    
    // Initialize resources
    void initialize();
    
    // Clean up resources
    void cleanup();
};

} // namespace gpu
} // namespace semantic_search