// src/benchmark_app.cpp
#include "utils/data_loader.h"
#include "utils/benchmark.h"
#include "cpu/brute_force.h"
#include "cpu/brute_force_omp.h"
#include "cpu/faiss_wrapper.h"
#include "gpu/cuda_search.h"
#include "gpu/rapids_wrapper.h"
#include <iostream>
#include <string>

using namespace semantic_search;
using namespace semantic_search::utils;
using namespace semantic_search::cpu;
using namespace semantic_search::gpu;

int main(int argc, char** argv) {
    // Parse command line arguments
    int datasetSize = 100000;
    int dimensions = 128;
    int numQueries = 100;
    int k = 10;
    
    if (argc > 1) {
        datasetSize = std::stoi(argv[1]);
    }
    
    if (argc > 2) {
        dimensions = std::stoi(argv[2]);
    }
    
    if (argc > 3) {
        numQueries = std::stoi(argv[3]);
    }
    
    if (argc > 4) {
        k = std::stoi(argv[4]);
    }
    
    std::cout << "Benchmarking with:" << std::endl;
    std::cout << "  Dataset size: " << datasetSize << std::endl;
    std::cout << "  Dimensions: " << dimensions << std::endl;
    std::cout << "  Queries: " << numQueries << std::endl;
    std::cout << "  k: " << k << std::endl;
    
    // Generate dataset
    std::cout << "Generating dataset..." << std::endl;
    auto dataset = DataLoader::generateSampleData(datasetSize, dimensions);
    
    // Generate queries
    std::cout << "Generating queries..." << std::endl;
    auto queries = DataLoader::generateSampleData(numQueries, dimensions);
    
    // Create benchmark
    Benchmark benchmark;
    
    // Add algorithms
    benchmark.addAlgorithm(std::make_shared<BruteForceSearch>());
    benchmark.addAlgorithm(std::make_shared<BruteForceSearchOMP>());
    
    // Add FAISS if available
    try {
        benchmark.addAlgorithm(std::make_shared<FaissIVFPQ>(100, 16, 8));
    } catch (const std::exception& e) {
        std::cerr << "Failed to add FAISS: " << e.what() << std::endl;
    }
    
    // Add CUDA if available
    try {
        benchmark.addAlgorithm(std::make_shared<CudaSearch>());
    } catch (const std::exception& e) {
        std::cerr << "Failed to add CUDA: " << e.what() << std::endl;
    }
    
    // Add RAPIDS if available
    try {
        benchmark.addAlgorithm(std::make_shared<RapidsSearch>());
    } catch (const std::exception& e) {
        std::cerr << "Failed to add RAPIDS: " << e.what() << std::endl;
    }
    
    // Set dataset and queries
    benchmark.setDataset(dataset);
    benchmark.setQueries(queries);
    
    // Set ground truth using brute force search
    std::cout << "Calculating ground truth..." << std::endl;
    auto exactAlgorithm = std::make_shared<BruteForceSearch>();
    benchmark.setGroundTruth(exactAlgorithm);
    
    // Run benchmarks
    std::cout << "Running benchmarks..." << std::endl;
    auto results = benchmark.runBenchmarks(k, numQueries);
    
    // Print results
    benchmark.printResults(results);
    
    return 0;
}