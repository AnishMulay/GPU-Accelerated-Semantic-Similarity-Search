# README.md

# Semantic Similarity Search with GPU Acceleration

Accelerated semantic similarity search that leverages CPU and GPU parallelization techniques. This project implements and compares five different approaches to vector similarity search, with a focus on performance optimization.

## 📋 Overview

This project investigates the performance of different semantic similarity search implementations across CPU and GPU architectures. By implementing and benchmarking multiple approaches, it provides insights into the scalability and efficiency of various parallelization techniques for high-dimensional vector operations.

### Key Features

- **Multiple Implementations**:
  - 🔍 CPU Brute Force (single-threaded baseline)
  - 🧵 CPU Brute Force with OpenMP (multi-threaded)
  - 📚 FAISS IVFPQ (optimized CPU with approximate nearest neighbor search)
  - 🚀 Custom CUDA implementation (GPU-accelerated brute force)
  - 🔥 RAPIDS cuML integration (state-of-the-art GPU library)

- **Comprehensive Benchmarking**:
  - ⏱️ Query latency (p50, p95, p99)
  - 📈 Throughput (queries per second)
  - 🎯 Recall@10 accuracy
  - 💾 Memory utilization

- **Optimized Vector Operations**:
  - Cosine similarity calculations
  - K-nearest neighbor search
  - Efficient vector data management

## 🧩 Project Structure

```
semantic_search_project/
├── CMakeLists.txt                # Main CMake configuration
├── src/
│   ├── main.cpp                  # Main executable
│   ├── benchmark_app.cpp         # Benchmarking application
│   ├── analysis_app.cpp          # Analysis application
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
│   ├── vector_data.h             # Vector data class
│   ├── utils/                    # Utility headers
│   ├── cpu/                      # CPU implementation headers
│   └── gpu/                      # GPU implementation headers
├── tests/                        # Test suite
│   ├── test_vector_data.cpp      # Vector data tests
│   ├── test_brute_force.cpp      # CPU tests
│   ├── test_cuda.cpp             # CUDA tests
│   └── test_faiss.cpp            # FAISS tests
├── data/                         # Data directory
│   └── README.md                 # Data description
├── scripts/                      # Helper scripts
├── build/                        # Build directory
├── INSTALL.md                    # Detailed installation guide
└── LICENSE                       # License information
```

## 🔬 Implementation Details

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
- **Memory Coalescing**: Ensuring that consecutive threads access consecutive memory locations
- **Hierarchical Parallelism**: Leveraging CUDA's thread/block hierarchy for maximum parallelism
- **Kernel Fusion**: Combining dot product, normalization, and division into a single operation

#### 2. RAPIDS cuML Integration

RAPIDS cuML provides GPU-accelerated machine learning algorithms, including nearest neighbor search. The implementation integrates with this library to leverage its optimized CUDA primitives and memory management.

RAPIDS cuML advantages:
- **Optimized CUDA Primitives**: Built on highly tuned libraries like cuBLAS and cuSparse
- **Memory Management**: Efficient GPU memory utilization
- **Advanced Algorithms**: GPU-optimized versions of complex nearest neighbor algorithms

## 📊 Performance Results

### Key Findings

- **GPU Acceleration**: My CUDA implementation achieves up to 20× speedup over the CPU baseline for small to medium datasets
- **OpenMP Scaling**: The OpenMP implementation provides 3-4× speedup on multi-core systems
- **FAISS Efficiency**: FAISS IVFPQ shows exceptional performance for large datasets, up to 782× faster than CPU baseline
- **Dimensionality Impact**: Performance degradation with increasing dimensions is much less severe in GPU implementations

### Sample Results (10,000 vectors, 128 dimensions)

| Algorithm | Avg (ms) | p99 (ms) | QPS | Recall@10 |
|-----------|----------|----------|-----|-----------|
| CPU Brute Force | 9.59 | 14.36 | 104.31 | 1.0000 |
| CPU Brute Force (OpenMP) | 5.47 | 12.90 | 182.86 | 1.0000 |
| FAISS IVFPQ | 0.82 | 9.57 | 1216.03 | 1.0000 |
| CUDA Cosine Similarity | 1.66 | 5.29 | 603.99 | 1.0000 |
| RAPIDS cuML | 0.87 | 2.65 | 1149.43 | 0.9890 |

For complete results across all dataset sizes and dimensions, see the [full benchmark report](https://github.com/yourusername/semantic-search-gpu/blob/main/docs/benchmarks.md).

## 🚀 Usage

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

### Simple Example

```cpp
#include "vector_data.h"
#include "cpu/brute_force.h"
#include "gpu/cuda_search.h"

int main() {
    // Generate sample data
    auto dataset = DataLoader::generateSampleData(10000, 128);
    
    // Create search algorithm
    auto search = std::make_shared();
    search->setDataset(dataset);
    
    // Prepare query vector
    std::vector query(128);
    // ... fill query vector ...
    
    // Search for nearest neighbors
    auto results = search->search(query, 10);
    
    // Print results
    for (const auto& result : results) {
        std::cout << "Index: " << result.index 
                  << ", Similarity: " << result.similarity 
                  << ", Label: " << result.label << std::endl;
    }
    
    return 0;
}
```

## 🔧 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- The instructor and teaching assistants of the CSC 548 Parallel Systems course for their guidance
- The FAISS team at Facebook Research for their excellent library
- The RAPIDS team at NVIDIA for their GPU-accelerated ML ecosystem

## 📚 References

- [FAISS GitHub Repository](https://github.com/facebookresearch/faiss)
- [RAPIDS cuML Documentation](https://docs.rapids.ai/api/cuml/stable/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [OpenMP Documentation](https://www.openmp.org/resources/)

---

For detailed installation instructions, please refer to [INSTALL.md](INSTALL.md).

# INSTALL.md

# Installation Guide

This guide provides detailed instructions for setting up the Semantic Similarity Search project, with special focus on installing the FAISS and RAPIDS cuML dependencies.

## Table of Contents

- [System Requirements](#system-requirements)
- [Basic Installation](#basic-installation)
- [CUDA Installation](#cuda-installation)
- [FAISS Installation](#faiss-installation)
  - [FAISS via Conda](#faiss-via-conda)
  - [FAISS via pip](#faiss-via-pip)
  - [FAISS from Source](#faiss-from-source)
- [RAPIDS cuML Installation](#rapids-cuml-installation)
  - [RAPIDS via Docker](#rapids-via-docker)
  - [RAPIDS via Conda](#rapids-via-conda)
  - [RAPIDS from Source](#rapids-from-source)
- [Project Build and Installation](#project-build-and-installation)
- [Troubleshooting](#troubleshooting)
- [Verifying Installation](#verifying-installation)

## System Requirements

### Minimum Requirements

- **CPU**: Modern x86_64 processor (Intel Core i5/i7 or AMD Ryzen)
- **RAM**: 8GB (16GB+ recommended for large datasets)
- **Storage**: 5GB free space
- **OS**: Ubuntu 18.04+, CentOS 7+, Windows 10, or macOS 10.14+
- **GPU** (for CUDA/RAPIDS): NVIDIA GPU with Compute Capability 6.0+ (Pascal architecture or newer)

### Recommended Configuration

- **CPU**: 8+ cores
- **RAM**: 32GB+
- **GPU**: NVIDIA GeForce RTX series or Tesla/A100 series
- **CUDA**: CUDA 11.x or 12.x

## Basic Installation

First, clone the repository and create a build directory:

```bash
git clone https://github.com/yourusername/semantic-search-gpu.git
cd semantic-search-gpu
mkdir build
```

### Required Dependencies

Install basic dependencies:

#### Ubuntu/Debian

```bash
sudo apt update
sudo apt install -y build-essential cmake git curl wget
sudo apt install -y libopenblas-dev liblapack-dev
```

#### CentOS/RHEL

```bash
sudo yum groupinstall -y "Development Tools"
sudo yum install -y cmake3 openblas-devel lapack-devel
```

#### macOS

```bash
brew install cmake openblas lapack libomp
```

#### Windows

Install [Visual Studio](https://visualstudio.microsoft.com/downloads/) with C++ development tools and [CMake](https://cmake.org/download/).

## CUDA Installation

CUDA is required for the GPU implementations. Skip this section if you only plan to use CPU implementations.

### Ubuntu

```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

# Install CUDA
sudo apt update
sudo apt install -y cuda-11-8 cuda-drivers
```

### CentOS/RHEL

```bash
sudo yum install -y https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-repo-rhel7-10.2.89-1.x86_64.rpm
sudo yum clean all
sudo yum install -y cuda-11-8
```

### Windows

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for Windows.

### Post-Installation

Add CUDA to your PATH:

```bash
# Add to ~/.bashrc (Linux) or ~/.bash_profile (macOS)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Verify the installation:

```bash
nvcc --version
```

## FAISS Installation

FAISS (Facebook AI Similarity Search) is a library for efficient similarity search. There are several ways to install it.

### FAISS via Conda

This is the easiest method if you use Conda:

```bash
# Create a new conda environment
conda create -n semantic_search python=3.8
conda activate semantic_search

# Install FAISS with GPU support
conda install -c pytorch faiss-gpu cudatoolkit=11.8

# For CPU-only
# conda install -c pytorch faiss-cpu
```

### FAISS via pip

For pip installation:

```bash
# GPU version
pip install faiss-gpu

# CPU version
# pip install faiss-cpu
```

### FAISS from Source

Building from source gives you the most control over optimization:

#### Prerequisites

```bash
sudo apt install -y swig libblas-dev liblapack-dev
```

#### Clone and Build

```bash
git clone https://github.com/facebookresearch/faiss.git
cd faiss

# Configure with GPU support
cmake -B build -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=ON -DCMAKE_BUILD_TYPE=Release

# For CPU only, use:
# cmake -B build -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release -j $(nproc)

# Install
sudo cmake --install build
```

#### Python Bindings (Optional)

```bash
cd build/faiss/python
pip install -e .
```

### Verifying FAISS Installation

To verify the C++ installation:

```bash
cd build
make test_blas
./misc/test_blas
```

For Python installation:

```python
import faiss
print(faiss.__version__)
# Check GPU availability
print("GPU available:", faiss.get_num_gpus())
```

## RAPIDS cuML Installation

RAPIDS is a suite of GPU-accelerated libraries for data science, including cuML for machine learning algorithms.

### RAPIDS via Docker

Docker is the simplest way to get started with RAPIDS:

```bash
# Pull the latest RAPIDS container
docker pull rapidsai/rapidsai:23.02-cuda11.8-runtime-ubuntu20.04-py3.9

# Run the container with GPU access
docker run --gpus all --rm -it \
    -p 8888:8888 -p 8787:8787 -p 8786:8786 \
    rapidsai/rapidsai:23.02-cuda11.8-runtime-ubuntu20.04-py3.9
```

### RAPIDS via Conda

```bash
# Create a new conda environment
conda create -n rapids-23.02 python=3.9
conda activate rapids-23.02

# Install RAPIDS
conda install -c rapidsai -c conda-forge -c nvidia \
    rapids=23.02 python=3.9 cudatoolkit=11.8
```

### RAPIDS from Source

Building RAPIDS from source is complex due to its many dependencies. The recommended approach is to use the Conda packages.

If you need to build from source:

1. Follow the [RAPIDS Build Guide](https://github.com/rapidsai/cuml/blob/branch-23.02/BUILD.md)
2. Build in stages (rmm → cudf → cuml)

```bash
# Example commands (see full guide for details)
git clone https://github.com/rapidsai/rmm.git
cd rmm
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install
make -j$(nproc) install

# Repeat similar steps for cudf and cuml
```

### Verifying RAPIDS Installation

To verify your RAPIDS cuML installation:

```python
import cuml
print(cuml.__version__)

# Try a simple operation
from cuml.neighbors import NearestNeighbors
import cupy as cp

# Create random data
X = cp.random.random((1000, 50))
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X)
distances, indices = nn.kneighbors(X[:10])
print(indices)
```

## Project Build and Installation

Once you have installed the dependencies, you can build the project:

```bash
cd semantic_search_gpu/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### CMake Options

You can customize the build with various CMake options:

```bash
# CPU-only build (no CUDA)
cmake .. -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release

# Specify CUDA architecture
cmake .. -DCUDA_ARCH=75 -DCMAKE_BUILD_TYPE=Release

# Enable FAISS
cmake .. -DUSE_FAISS=ON -DCMAKE_BUILD_TYPE=Release

# Enable RAPIDS (requires CUDA)
cmake .. -DUSE_RAPIDS=ON -DCMAKE_BUILD_TYPE=Release

# Specify custom install location
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install -DCMAKE_BUILD_TYPE=Release
```

### Installation

```bash
# Install to system (may require sudo)
make install
```

## Troubleshooting

### Common Issues

#### CUDA Not Found

If CMake can't find CUDA:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

#### FAISS Linking Issues

If you get linker errors with FAISS:

```bash
# Make sure pkgconfig can find FAISS
export PKG_CONFIG_PATH=/path/to/faiss/installation/lib/pkgconfig:$PKG_CONFIG_PATH
```

#### RAPIDS Import Errors

If you see import errors with RAPIDS:

```bash
# Check PYTHONPATH
export PYTHONPATH=/path/to/rapids/installation/python:$PYTHONPATH
```

### Platform-Specific Issues

#### Windows CUDA Issues

On Windows, you may need to specify the CUDA toolkit directory explicitly:

```bash
cmake .. -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin/nvcc.exe"
```

#### macOS OpenMP Issues

On macOS with Apple clang, OpenMP support requires extra flags:

```bash
cmake .. -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp" -DOpenMP_CXX_LIB_NAMES="omp" -DOpenMP_omp_LIBRARY=/usr/local/opt/libomp/lib/libomp.dylib
```

## Verifying Installation

### Run Tests

After building, run the test suite:

```bash
cd build
make test
# Or run specific tests
./tests/test_vector_data
./tests/test_brute_force
./tests/test_cuda
./tests/test_faiss
```

### Run Benchmarks

```bash
# Run with default parameters
./benchmark_app

# Run with custom parameters
./benchmark_app 10000 128 100 10
```

### Expected Output

The benchmarking application should produce output similar to:

```
=== Benchmark Results ===
Algorithm                Avg Time (ms)  p50 (ms)       p95 (ms)       p99 (ms)       QPS            Recall@10      
------------------------------------------------------------------------------------------------------------------------
CPU Brute Force          9.59           9.01           13.20          14.36          104.31         1.0000         
CPU Brute Force (OpenMP) 5.47           5.11           9.36           12.90          182.86         1.0000         
FAISS IVFPQ              0.82           0.51           2.24           9.57           1216.03        1.0000         
CUDA Cosine Similarity   1.66           1.56           2.24           5.29           603.99         1.0000         
RAPIDS cuML              0.87           0.81           1.18           2.65           1149.43        0.9890
```

This indicates that all implementations are working correctly.

---

If you encounter any issues during installation or have questions, please [open an issue](https://github.com/yourusername/semantic-search-gpu/issues) on the GitHub repository.

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/33272401/a7b3a982-ea1e-48de-a5c5-889919dae056/amulay2_CSC_548-1.pdf

---
Answer from Perplexity: pplx.ai/share
