#pragma once
#include "similarity_search.h"
#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/device_mdarray.hpp>

namespace semantic_search {
namespace gpu {

class RapidsSearch : public SimilaritySearch {
public:
    RapidsSearch();
    ~RapidsSearch() override;
    
    void setDataset(VectorDataPtr dataset) override;
    std::vector<SearchResult> search(const std::vector<float>& query, int k) override;
    std::string getName() const override { return "RAPIDS cuML"; }

private:
    VectorDataPtr m_dataset;
    raft::device_resources m_res;
    std::unique_ptr<cuvs::neighbors::ivf_pq::index<int64_t>> m_index;
    int64_t m_dimensions;
};

} // namespace gpu
} // namespace semantic_search