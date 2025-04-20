// include/cpu/faiss_wrapper.h
#pragma once
#include "similarity_search.h"
#include <vector>
#include <string>
#include <memory>

// Forward declaration to avoid including FAISS headers here
namespace faiss {
    class Index;
    class IndexIVFPQ;
}

namespace semantic_search {
namespace cpu {

class FaissIVFPQ : public SimilaritySearch {
public:
    FaissIVFPQ(int numLists = 100, int numCodes = 16, int bitsPerCode = 8);
    ~FaissIVFPQ() override;
    
    // Initialize with a dataset
    void setDataset(VectorDataPtr dataset) override;
    
    // Search for nearest neighbors
    std::vector<SearchResult> search(const std::vector<float>& query, int k) override;
    
    // Get the name of the algorithm
    std::string getName() const override { return "FAISS IVFPQ"; }
    
    // Set number of lists to probe during search
    void setNProbe(int nprobe);
    
private:
    VectorDataPtr m_dataset;
    std::unique_ptr<faiss::IndexIVFPQ> m_index;
    int m_numLists;
    int m_numCodes;
    int m_bitsPerCode;
    int m_nprobe;
    bool m_trained;
    
    // Train the index
    void trainIndex();
};

} // namespace cpu
} // namespace semantic_search