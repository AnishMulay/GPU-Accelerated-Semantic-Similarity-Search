// src/utils/benchmark.cpp
#include "utils/benchmark.h"
#include "utils/timer.h"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <cstdlib>

namespace semantic_search {
namespace utils {

Benchmark::Benchmark() {
}

void Benchmark::addAlgorithm(SimilaritySearchPtr algorithm) {
    m_algorithms.push_back(algorithm);
}

void Benchmark::setDataset(VectorDataPtr dataset) {
    m_dataset = dataset;
    
    // Set dataset for all algorithms
    for (auto& algorithm : m_algorithms) {
        algorithm->setDataset(dataset);
    }
}

void Benchmark::setQueries(VectorDataPtr queries) {
    m_queries = queries;
}

void Benchmark::setGroundTruth(SimilaritySearchPtr exactAlgorithm) {
    if (!m_dataset || !m_queries) {
        throw std::runtime_error("Dataset and queries must be set before ground truth");
    }
    
    // Set the dataset for the exact algorithm
    exactAlgorithm->setDataset(m_dataset);
    
    // Calculate ground truth for each query
    std::cout << "Calculating ground truth..." << std::endl;
    m_groundTruth.clear();
    
    size_t numQueries = m_queries->getSize();
    m_groundTruth.resize(numQueries);
    
    for (size_t i = 0; i < numQueries; i++) {
        const float* queryVec = m_queries->getVector(i);
        std::vector<float> query(m_queries->getDimensions());
        std::copy(queryVec, queryVec + m_queries->getDimensions(), query.begin());
        
        // Get exact results
        auto results = exactAlgorithm->search(query, m_k);
        
        // Extract indices
        auto& groundTruth = m_groundTruth[i];
        groundTruth.resize(results.size());
        
        for (size_t j = 0; j < results.size(); j++) {
            groundTruth[j] = results[j].index;
        }
    }
    
    std::cout << "Ground truth calculated for " << numQueries << " queries" << std::endl;
}

std::vector<BenchmarkResult> Benchmark::runBenchmarks(int k, int numQueries) {
    if (!m_dataset || !m_queries) {
        throw std::runtime_error("Dataset and queries must be set before benchmarking");
    }
    
    m_k = k;
    
    // Adjust number of queries if necessary
    numQueries = std::min(numQueries, static_cast<int>(m_queries->getSize()));
    
    std::vector<BenchmarkResult> results;
    
    // Benchmark each algorithm
    for (auto& algorithm : m_algorithms) {
        std::cout << "Benchmarking " << algorithm->getName() << "..." << std::endl;
        
        BenchmarkResult result;
        result.algorithmName = algorithm->getName();
        
        // Prepare for benchmark
        std::vector<double> queryTimes;
        std::vector<std::vector<uint32_t>> allResults;
        
        // Run queries
        for (int i = 0; i < numQueries; i++) {
            const float* queryVec = m_queries->getVector(i);
            std::vector<float> query(m_queries->getDimensions());
            std::copy(queryVec, queryVec + m_queries->getDimensions(), query.begin());
            
            // Measure search time
            Timer timer("Search");
            auto searchResults = algorithm->search(query, k);
            double queryTime = timer.elapsedMilliseconds();
            
            // Record time
            queryTimes.push_back(queryTime);
            
            // Record results for recall calculation
            if (!m_groundTruth.empty()) {
                std::vector<uint32_t> resultIndices;
                resultIndices.reserve(searchResults.size());
                
                for (const auto& result : searchResults) {
                    resultIndices.push_back(result.index);
                }
                
                allResults.push_back(resultIndices);
            }
        }
        
        // Calculate metrics
        result.avgQueryTime = std::accumulate(queryTimes.begin(), queryTimes.end(), 0.0) / queryTimes.size();
        result.throughput = 1000.0 / result.avgQueryTime; // Queries per second
        
        // Calculate latency percentiles
        std::sort(queryTimes.begin(), queryTimes.end());
        result.p50Latency = queryTimes[queryTimes.size() * 0.5];
        result.p95Latency = queryTimes[queryTimes.size() * 0.95];
        result.p99Latency = queryTimes[queryTimes.size() * 0.99];
        
        // Calculate recall if ground truth is available
        if (!m_groundTruth.empty()) {
            float totalRecall = 0.0f;
            
            for (int i = 0; i < numQueries; i++) {
                totalRecall += calculateRecall(m_groundTruth[i], allResults[i]);
            }
            
            result.recall = totalRecall / numQueries;
        } else {
            result.recall = -1.0f; // Not available
        }
        
        // Estimate memory usage
        result.memoryUsage = estimateMemoryUsage(algorithm);
        
        // Add to results
        results.push_back(result);
    }
    
    return results;
}

void Benchmark::printResults(const std::vector<BenchmarkResult>& results) {
    // Print header
    std::cout << "\n=== Benchmark Results ===" << std::endl;
    std::cout << std::left << std::setw(25) << "Algorithm" 
              << std::setw(15) << "Avg Time (ms)" 
              << std::setw(15) << "p50 (ms)" 
              << std::setw(15) << "p95 (ms)" 
              << std::setw(15) << "p99 (ms)" 
              << std::setw(15) << "QPS" 
              << std::setw(15) << "Recall@" << m_k;
    
    if (results[0].memoryUsage > 0) {
        std::cout << std::setw(15) << "Memory (MB)";
    }
    
    std::cout << std::endl;
    
    // Print divider
    std::cout << std::string(120, '-') << std::endl;
    
    // Print results
    for (const auto& result : results) {
        std::cout << std::left << std::setw(25) << result.algorithmName 
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.avgQueryTime 
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.p50Latency 
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.p95Latency 
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.p99Latency 
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.throughput;
        
        if (result.recall >= 0.0f) {
            std::cout << std::setw(15) << std::fixed << std::setprecision(4) << result.recall;
        } else {
            std::cout << std::setw(15) << "N/A";
        }
        
        if (result.memoryUsage > 0) {
            std::cout << std::setw(15) << std::fixed << std::setprecision(2) << result.memoryUsage / (1024.0 * 1024.0);
        }
        
        std::cout << std::endl;
    }
}

float Benchmark::calculateRecall(const std::vector<uint32_t>& groundTruth, 
                               const std::vector<uint32_t>& results) {
    if (groundTruth.empty() || results.empty()) {
        return 0.0f;
    }
    
    // Count the number of results that are in the ground truth
    int numCorrect = 0;
    
    for (uint32_t resultIdx : results) {
        if (std::find(groundTruth.begin(), groundTruth.end(), resultIdx) != groundTruth.end()) {
            numCorrect++;
        }
    }
    
    // Calculate recall
    return static_cast<float>(numCorrect) / std::min(groundTruth.size(), results.size());
}

size_t Benchmark::estimateMemoryUsage(SimilaritySearchPtr algorithm) {
    // This is a very crude estimation
    // In a real implementation, you would want to use a profiling tool
    
    // Base memory usage for vectors
    size_t baseMemory = 0;
    if (m_dataset) {
        baseMemory = m_dataset->getDimensions() * m_dataset->getSize() * sizeof(float);
    }
    
    // Add algorithm-specific overhead (very rough estimates)
    if (algorithm->getName().find("FAISS") != std::string::npos) {
        // FAISS typically uses additional memory for indexes
        baseMemory = baseMemory * 1.5;
    } else if (algorithm->getName().find("CUDA") != std::string::npos ||
               algorithm->getName().find("RAPIDS") != std::string::npos) {
        // GPU implementations may have duplicated data
        baseMemory = baseMemory * 2.0;
    }
    
    return baseMemory;
}

} // namespace utils
} // namespace semantic_search