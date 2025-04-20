// include/utils/benchmark.h
#pragma once
#include "similarity_search.h"
#include "vector_data.h"
#include <vector>
#include <string>
#include <memory>
#include <functional>

namespace semantic_search {
namespace utils {

struct BenchmarkResult {
    std::string algorithmName;
    double avgQueryTime;
    double p50Latency;
    double p95Latency;
    double p99Latency;
    double throughput;
    float recall;
    size_t memoryUsage;
};

class Benchmark {
public:
    Benchmark();
    
    // Add an algorithm to benchmark
    void addAlgorithm(SimilaritySearchPtr algorithm);
    
    // Set the dataset
    void setDataset(VectorDataPtr dataset);
    
    // Set the queries
    void setQueries(VectorDataPtr queries);
    
    // Set ground truth (from an exact algorithm)
    void setGroundTruth(SimilaritySearchPtr exactAlgorithm);
    
    // Run benchmarks
    std::vector<BenchmarkResult> runBenchmarks(int k, int numQueries = 100);
    
    // Print results
    void printResults(const std::vector<BenchmarkResult>& results);
    
private:
    std::vector<SimilaritySearchPtr> m_algorithms;
    VectorDataPtr m_dataset;
    VectorDataPtr m_queries;
    std::vector<std::vector<uint32_t>> m_groundTruth;
    int m_k;
    
    // Calculate recall@k
    float calculateRecall(const std::vector<uint32_t>& groundTruth, 
                         const std::vector<uint32_t>& results);
                         
    // Estimate memory usage
    size_t estimateMemoryUsage(SimilaritySearchPtr algorithm);
};

} // namespace utils
} // namespace semantic_search