// include/gpu/cuda_kernels.h
#pragma once
#include <cuda_runtime.h>
#include <vector>

namespace semantic_search {
namespace gpu {

// Launch cosine similarity kernel
void launchCosineSimilarityKernel(const float* vectors, const float* query, 
                                float* similarities, int numVectors, int dimensions, 
                                cudaStream_t stream = 0);

// Copy vectors to device
void copyVectorsToDevice(const float* vectors, float** deviceVectors, 
                        size_t numVectors, size_t dimensions);

// Copy query to device
void copyQueryToDevice(const float* query, float** deviceQuery, size_t dimensions);

// Free device memory
void freeDeviceMemory(void* devicePtr);

} // namespace gpu
} // namespace semantic_search