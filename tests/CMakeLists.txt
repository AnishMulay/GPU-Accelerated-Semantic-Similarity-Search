add_executable(test_similarity test_similarity.cpp)
target_link_libraries(test_similarity PRIVATE 
    similarity_utils
    cpu_implementations
    gpu_implementations
    ${CUDA_LIBRARIES}
)

add_executable(test_vector_data test_vector_data.cpp)
target_link_libraries(test_vector_data PRIVATE 
    similarity_utils
    ${CUDA_LIBRARIES}
)

add_executable(test_brute_force test_brute_force.cpp)
target_link_libraries(test_brute_force PRIVATE 
    similarity_utils
    cpu_implementations
    ${CUDA_LIBRARIES}
)