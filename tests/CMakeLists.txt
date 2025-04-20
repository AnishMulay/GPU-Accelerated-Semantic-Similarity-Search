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

add_executable(test_brute_force_comparison test_brute_force_comparison.cpp)
target_link_libraries(test_brute_force_comparison PRIVATE 
    similarity_utils
    cpu_implementations
    ${CUDA_LIBRARIES}
    OpenMP::OpenMP_CXX
)

add_executable(test_faiss test_faiss.cpp)
target_link_libraries(test_faiss PRIVATE 
    similarity_utils
    cpu_implementations
    -lfaiss  # Direct link by name
    -lopenblas
    -llapack
    ${CUDA_LIBRARIES}
)

# tests/CMakeLists.txt
add_executable(test_cuda test_cuda.cpp)
target_link_libraries(test_cuda PRIVATE 
    similarity_utils
    cpu_implementations
    gpu_implementations
    ${CUDA_LIBRARIES}
)

# tests/CMakeLists.txt
add_executable(test_rapids test_rapids.cpp)
target_link_libraries(test_rapids PRIVATE 
    similarity_utils
    gpu_implementations
    ${CUDA_LIBRARIES}
    ${RapidsCuML_LIBRARIES}
)