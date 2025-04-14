// include/utils/metrics.h
#pragma once
#include <vector>
#include <cstdint>

namespace semantic_search {
namespace utils {

class Metrics {
public:
    // Calculate recall@k
    static float calculateRecall(const std::vector<uint32_t>& groundTruth, 
                               const std::vector<uint32_t>& results,
                               int k);
                               
    // Calculate query latency statistics
    static void calculateLatencyStats(const std::vector<double>& latencies,
                                    double& p50, double& p95, double& p99);
                                    
    // Calculate throughput (queries per second)
    static double calculateThroughput(int numQueries, double totalTime);
};

} // namespace utils
} // namespace semantic_search