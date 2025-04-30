#include "gpu/rapids_wrapper.h"
#include <raft/core/device_mdarray.hpp>

namespace semantic_search {
namespace gpu {

RapidsSearch::RapidsSearch() = default;
RapidsSearch::~RapidsSearch() = default;

void RapidsSearch::setDataset(VectorDataPtr dataset) {
    m_dataset = dataset;
    m_dimensions = dataset->getDimensions();

    // Create IVF-PQ index parameters
    cuvs::neighbors::ivf_pq::index_params index_params;
    index_params.n_lists = 100;

    // Create device matrix view
    auto dataset_view = raft::make_device_matrix_view<const float, int64_t>(
        m_dataset->getVectorsData(),
        m_dataset->getSize(),
        m_dimensions
    );

    // Build index
    m_index = std::make_unique<cuvs::neighbors::ivf_pq::index<int64_t>>(
        cuvs::neighbors::ivf_pq::build(m_res, index_params, dataset_view)
    );
}

std::vector<SearchResult> RapidsSearch::search(const std::vector<float>& query, int k) {
    if (!m_index || query.size() != static_cast<size_t>(m_dimensions)) {
        throw std::runtime_error("Invalid search parameters");
    }

    // Create query view
    auto query_view = raft::make_device_matrix_view<const float, int64_t>(
        query.data(),
        1,  // Single query
        m_dimensions
    );

    // Prepare output buffers
    auto indices = raft::make_device_matrix<int64_t, int64_t>(m_res, 1, k);
    auto distances = raft::make_device_matrix<float, int64_t>(m_res, 1, k);

    // Configure search
    cuvs::neighbors::ivf_pq::search_params search_params;
    search_params.n_probes = 20;

    // Execute search
    cuvs::neighbors::ivf_pq::search(
        m_res,
        search_params,
        *m_index,
        query_view,
        indices.view(),
        distances.view()
    );

    // Copy results to host
    std::vector<int64_t> host_indices(k);
    std::vector<float> host_distances(k);
    
    raft::copy(host_indices.data(), indices.data_handle(), k, m_res.get_stream());
    raft::copy(host_distances.data(), distances.data_handle(), k, m_res.get_stream());

    // Convert to search results
    std::vector<SearchResult> results;
    for (int i = 0; i < k; i++) {
        if (host_indices[i] >= 0) {
            results.emplace_back(
                static_cast<uint32_t>(host_indices[i]),
                1.0f / (1.0f + host_distances[i]),
                m_dataset->getLabel(host_indices[i])
            );
        }
    }
    
    return results;
}

} // namespace gpu
} // namespace semantic_search