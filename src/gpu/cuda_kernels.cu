// src/gpu/cuda_kernels.cu
#include "gpu/cuda_kernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <iostream>

namespace semantic_search {
namespace gpu {

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(error) << std::endl; \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// CUDA kernel for cosine similarity calculation
__global__ void cosineSimilarityKernel(const float* vectors, const float* query, 
                                      float* similarities, int numVectors, int dimensions) {
    int vectorIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (vectorIdx < numVectors) {
        float dotProduct = 0.0f;
        float normVector = 0.0f;
        float normQuery = 0.0f;
        
        for (int i = 0; i < dimensions; i++) {
            float vectorVal = vectors[vectorIdx * dimensions + i];
            float queryVal = query[i];
            
            dotProduct += vectorVal * queryVal;
            normVector += vectorVal * vectorVal;
            normQuery += queryVal * queryVal;
        }
        
        // Avoid division by zero
        if (normVector == 0.0f || normQuery == 0.0f) {
            similarities[vectorIdx] = 0.0f;
        } else {
            similarities[vectorIdx] = dotProduct / (sqrtf(normVector) * sqrtf(normQuery));
        }
    }
}

// Launch the cosine similarity kernel
void launchCosineSimilarityKernel(const float* vectors, const float* query, 
                                float* similarities, int numVectors, int dimensions, 
                                cudaStream_t stream) {
    // Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (numVectors + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernel
    cosineSimilarityKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        vectors, query, similarities, numVectors, dimensions);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for kernel to complete (if not using streams)
    if (stream == 0) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// Copy vectors to device
void copyVectorsToDevice(const float* vectors, float** deviceVectors, 
                        size_t numVectors, size_t dimensions) {
    size_t vectorsSize = numVectors * dimensions * sizeof(float);
    CUDA_CHECK(cudaMalloc(deviceVectors, vectorsSize));
    CUDA_CHECK(cudaMemcpy(*deviceVectors, vectors, vectorsSize, cudaMemcpyHostToDevice));
}

// Copy query to device
void copyQueryToDevice(const float* query, float** deviceQuery, size_t dimensions) {
    size_t querySize = dimensions * sizeof(float);
    CUDA_CHECK(cudaMalloc(deviceQuery, querySize));
    CUDA_CHECK(cudaMemcpy(*deviceQuery, query, querySize, cudaMemcpyHostToDevice));
}

// Free device memory
void freeDeviceMemory(void* devicePtr) {
    if (devicePtr != nullptr) {
        CUDA_CHECK(cudaFree(devicePtr));
    }
}

} // namespace gpu
} // namespace semantic_search