# Semantic Similarity Search with GPU Acceleration

## Overview

This project investigates the performance of different semantic similarity search implementations across CPU and GPU architectures. By implementing and benchmarking multiple approaches, it provides insights into the scalability and efficiency of various parallelization techniques for high-dimensional vector operations.

### Key Features

- **Multiple Implementations**:
  - CPU Brute Force (single-threaded baseline)
  - CPU Brute Force with OpenMP (multi-threaded)
  - FAISS IVFPQ (optimized CPU with approximate nearest neighbor search)
  - Custom CUDA implementation (GPU-accelerated brute force)
  - RAPIDS cuML integration (state-of-the-art GPU library)

- **Comprehensive Benchmarking**:
  - Query latency (p50, p95, p99)
  - Throughput (queries per second)
  - Recall@10 accuracy
  - Memory utilization

## Project Structure

```
GPU-Accelerated-Semantic-Similarity-Search/
├── CMakeLists.txt                # Main CMake configuration
├── src/
│   ├── main.cpp                  # Main executable
│   ├── benchmark_app.cpp         # Benchmarking application
│   ├── utils/                    # Utility classes
│   │   ├── data_loader.cpp       # Vector data loading
│   │   ├── timer.cpp             # Performance timing
│   │   ├── metrics.cpp           # Evaluation metrics
│   │   ├── vector_data.cpp       # Vector data management
│   │   └── benchmark.cpp         # Benchmarking utilities
│   ├── cpu/                      # CPU implementations
│   │   ├── brute_force.cpp       # Single-threaded baseline
│   │   ├── brute_force_omp.cpp   # OpenMP-accelerated
│   │   └── faiss_wrapper.cpp     # FAISS IVFPQ wrapper
│   └── gpu/                      # GPU implementations
│       ├── cuda_kernels.cu       # CUDA kernels
│       ├── cuda_search.cpp       # CUDA search implementation
│       └── rapids_wrapper.cpp    # RAPIDS cuML wrapper
├── include/                      # Header files
│   ├── similarity_search.h       # Base interface
│   ├── utils/                    # Utility headers
│   ├── cpu/                      # CPU implementation headers
│   └── gpu/                      # GPU implementation headers
├── tests/                        # Test suite
│   ├── test_vector_data.cpp      # Vector data tests
│   ├── test_similarity.cpp       # Similarity/Correctness tests
│   ├── test_brute_force.cpp      # CPU tests
│   ├── test_cuda.cpp             # CUDA tests
│   └── test_faiss.cpp            # FAISS tests
│   ├── test_rapids.cpp           # RAPIS CUML tests
├── build/                        # Build directory
└── LICENSE                       # License information
```

## Implementation Details

### Vector Data Management

The project uses a custom `VectorData` class that efficiently manages high-dimensional vector datasets:

```cpp
class VectorData {
public:
    VectorData(size_t dimensions);
    
    // Add a vector to the dataset
    void addVector(const std::vector& vector, 
                  const std::string& label = "");
    
    // Access a specific vector
    const float* getVector(size_t index) const;
    
    // Get all vectors as a contiguous array
    const float* getVectorsData() const;
    
private:
    size_t m_dimensions;
    std::vector m_vectors;  // Flattened for better memory layout
    std::vector m_labels;
};
```

Key optimization: Storing vectors in a flattened array rather than as separate objects improves cache locality and memory access patterns.

### CPU Implementations

#### 1. Baseline CPU (Brute Force)

The baseline implementation computes cosine similarity between the query vector and each database vector sequentially:
```cpp
float cosineSimilarity(const float* a, const float* b, size_t dimensions) {
    float dotProduct = 0.0f;
    float normA = 0.0f;
    float normB = 0.0f;
    
    for (size_t i = 0; i < dimensions; i++) {
        dotProduct += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    
    return dotProduct / (std::sqrt(normA) * std::sqrt(normB));
}

```

#### 2. OpenMP
The OpenMP implementation parallelizes the similarity calculations across available CPU cores:

```cpp
float cosineSimilarity(const float* a, const float* b, size_t dimensions) {
    float dotProduct = 0.0f;
    float normA = 0.0f;
    float normB = 0.0f;
    
    for (size_t i = 0; i  search(const std::vector& query, int k) {
    std::vector results(m_dataset->getSize());
    
    // Calculate similarity in parallel
    #pragma omp parallel for
    for (size_t i = 0; i getSize(); i++) {
        const float* vector = m_dataset->getVector(i);
        float similarity = cosineSimilarity(query.data(), vector, 
                                           m_dataset->getDimensions());
        results[i] = SearchResult(i, similarity, m_dataset->getLabel(i));
    }
    
    // Sort and return top-k results
    std::partial_sort(results.begin(), 
                     results.begin() + k, 
                     results.end());
    
    return std::vector(results.begin(), 
                                    results.begin() + k);
}
```

This implementation features:
- Parallel execution across CPU cores
- Loop unrolling for better instruction-level parallelism
- Pre-allocation of result vectors to avoid thread synchronization issues

#### 3. FAISS IVFPQ

FAISS (Facebook AI Similarity Search) uses a combination of techniques to accelerate similarity search:

- **Inverted File Index (IVF)**: Clusters database vectors and only searches within relevant clusters
- **Product Quantization (PQ)**: Compresses vectors by dividing them into subvectors and quantizing each separately

The implementation integrates with the FAISS C++ library to leverage these optimizations.

### GPU Implementations

#### 1. Custom CUDA Implementation

The heart of the GPU implementation is a set of custom CUDA kernels for similarity search operations:

```cuda
__global__ void cosineSimilarityKernel(
    const float* vectors,     // [n × d] database vectors
    const float* query,       // [d] query vector
    float* similarities,      // [n] similarity scores
    int numVectors,           // number of database vectors
    int dimensions)           // dimensions
{
    int vectorIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (vectorIdx  0.0f && normQuery > 0.0f) {
            similarities[vectorIdx] = dotProduct / 
                                    (sqrtf(normVector) * sqrtf(normQuery));
        } else {
            similarities[vectorIdx] = 0.0f;
        }
    }
}
```

Key optimizations include:
- **Shared Memory Caching**: Loading the query vector into fast shared memory once per thread block
- **Hierarchical Parallelism**: Leveraging CUDA's thread/block hierarchy for maximum parallelism
- **Kernel Fusion**: Combining dot product, normalization, and division into a single operation

#### 2. RAPIDS cuML Integration

RAPIDS cuML provides GPU-accelerated machine learning algorithms, including nearest neighbor search. The implementation integrates with this library to leverage its optimized CUDA primitives and memory management.

RAPIDS cuML advantages:
- **Optimized CUDA Primitives**: Built on highly tuned libraries like cuBLAS and cuSparse
- **Memory Management**: Efficient GPU memory utilization
- **Advanced Algorithms**: GPU-optimized versions of complex nearest neighbor algorithms

## Performance Results

### Key Findings

- **GPU Acceleration**: My CUDA implementation achieves up to 10× speedup over the CPU baseline for small to medium datasets
- **OpenMP Scaling**: The OpenMP implementation provides 3-4× speedup on multi-core systems (I tested on my laptop which has 4 cores)
- **FAISS Efficiency**: FAISS IVFPQ shows exceptional performance for large datasets, much faster than CPU baseline
- **Dimensionality Impact**: Performance degradation with increasing dimensions is much less severe in GPU implementations

### Sample Results (10,000 vectors, 128 dimensions)

| Algorithm | Avg (ms) | p99 (ms) | QPS | Recall@10 |
|-----------|----------|----------|-----|-----------|
| CPU Brute Force | 9.59 | 14.36 | 104.31 | 1.0000 |
| CPU Brute Force (OpenMP) | 5.47 | 12.90 | 182.86 | 1.0000 |
| FAISS IVFPQ | 0.82 | 9.57 | 1216.03 | 1.0000 |
| CUDA Cosine Similarity | 1.66 | 5.29 | 603.99 | 1.0000 |
| RAPIDS cuML | 0.87 | 2.65 | 1149.43 | 0.9890 |

## Usage

### Building the Project

```bash
mkdir build && cd build
cmake ..
make
```

### Running the Benchmarks

```bash
# Run all benchmarks
./benchmark_app

# Specify dataset size, dimensions, queries, and k
./benchmark_app 100000 128 1000 10
```

### Running the Tests

```bash
# Run all tests
ctest

# Run specific test
./tests/test_cuda
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The instructor and teaching assistants of the CSC 548 Parallel Systems course for their guidance
- The FAISS team at Facebook Research for their excellent library
- The RAPIDS team at NVIDIA for their GPU-accelerated ML ecosystem

## References

- [FAISS GitHub Repository](https://github.com/facebookresearch/faiss)
- [RAPIDS cuML Documentation](https://docs.rapids.ai/api/cuml/stable/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [OpenMP Documentation](https://www.openmp.org/resources/)
