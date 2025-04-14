add_executable(test_similarity test_similarity.cpp)
target_link_libraries(test_similarity PRIVATE 
    similarity_utils
    cpu_implementations
    gpu_implementations
    ${CUDA_LIBRARIES}
)