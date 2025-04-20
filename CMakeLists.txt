cmake_minimum_required(VERSION 3.14)
project(SemanticSearch LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find FAISS
find_package(faiss REQUIRED)

# Find BLAS and LAPACK (required by FAISS)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Try to find MKL (preferred for FAISS)
find_package(MKL QUIET)

# Output found libraries for debugging
message(STATUS "FAISS_LIBRARIES: ${FAISS_LIBRARIES}")
message(STATUS "BLAS_LIBRARIES: ${BLAS_LIBRARIES}")
message(STATUS "LAPACK_LIBRARIES: ${LAPACK_LIBRARIES}")

link_directories(/usr/local/lib)

# Modern way to find CUDA
find_package(CUDAToolkit REQUIRED)

# Find required packages
find_package(OpenMP REQUIRED)

# Add include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(SYSTEM /usr/include /usr/local/include)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)

# Create executable
add_executable(semantic_search src/main.cpp)
target_link_libraries(semantic_search PRIVATE 
    similarity_utils
    cpu_implementations
    gpu_implementations
    CUDA::cudart
    OpenMP::OpenMP_CXX
)