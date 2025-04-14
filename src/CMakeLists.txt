# Libraries
add_library(similarity_utils 
    utils/data_loader.cpp
    utils/metrics.cpp
    utils/timer.cpp
)

add_library(cpu_implementations
    cpu/brute_force.cpp
    cpu/faiss_wrapper.cpp
)

add_library(gpu_implementations
    gpu/cuda_kernels.cu
    gpu/rapids_wrapper.cpp
)

# Set include directories for libraries
target_include_directories(similarity_utils PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(cpu_implementations PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(gpu_implementations PUBLIC ${PROJECT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS})

# Link with CUDA
target_link_libraries(gpu_implementations PRIVATE ${CUDA_LIBRARIES})