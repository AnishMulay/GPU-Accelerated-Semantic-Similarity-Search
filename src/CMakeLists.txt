# Libraries
# Libraries
add_library(similarity_utils 
    utils/data_loader.cpp
    utils/metrics.cpp
    utils/timer.cpp
    utils/vector_data.cpp
    utils/benchmark.cpp  # Add this line
)

add_library(cpu_implementations
    cpu/brute_force.cpp
    cpu/brute_force_omp.cpp
    cpu/faiss_wrapper.cpp
)

add_library(gpu_implementations
    gpu/cuda_kernels.cu
    gpu/cuda_search.cpp
    gpu/rapids_wrapper.cpp
)

# Link CPU implementations with FAISS
# Link CPU implementations with FAISS and its dependencies
target_link_libraries(cpu_implementations PRIVATE 
    OpenMP::OpenMP_CXX
    faiss
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

# Set include directories for libraries
target_include_directories(similarity_utils PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(cpu_implementations PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(cpu_implementations PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(gpu_implementations PUBLIC ${PROJECT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS})
target_link_libraries(gpu_implementations PRIVATE ${CUDA_LIBRARIES})

set_target_properties(gpu_implementations PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)

# Link with CUDA
target_link_libraries(gpu_implementations PRIVATE ${CUDA_LIBRARIES})

# Create benchmarking application
add_executable(benchmark_app benchmark_app.cpp)
target_link_libraries(benchmark_app PRIVATE 
    similarity_utils
    cpu_implementations
    gpu_implementations
    ${CUDA_LIBRARIES}
    ${Faiss_LIBRARIES}
    ${RapidsCuML_LIBRARIES}
    OpenMP::OpenMP_CXX
)