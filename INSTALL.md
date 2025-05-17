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

If you encounter any issues during installation or have questions, please open an issue on this Github repo.
