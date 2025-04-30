cmake_minimum_required(VERSION 3.14)

# Set architecture before project()
set(CMAKE_CUDA_ARCHITECTURES "native")  # Auto-detect GPU arch

project(SemanticSearch LANGUAGES CXX CUDA)

# Before any include_directories() calls
list(REMOVE_ITEM CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
list(REMOVE_ITEM CMAKE_LIBRARY_PATH "$ENV{CONDA_PREFIX}/lib")

# After project() declaration
find_package(dlpack REQUIRED)
include_directories(${dlpack_INCLUDE_DIRS})

# Set standards before finding packages
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA first
find_package(CUDAToolkit REQUIRED)

# Configure RAFT components before finding
set(RAFT_COMPONENTS compiled)
find_package(raft REQUIRED COMPONENTS ${RAFT_COMPONENTS})

# Find FAISS with proper case
find_package(FAISS REQUIRED)

# BLAS/LAPACK configuration
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(MKL QUIET)  # Optional MKL

# OpenMP configuration
find_package(OpenMP REQUIRED)

# Critical path adjustments to avoid conda conflicts
list(REMOVE_ITEM CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
list(REMOVE_ITEM CMAKE_LIBRARY_PATH "$ENV{CONDA_PREFIX}/lib")

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${RAFT_INCLUDE_DIRS}
    ${FAISS_INCLUDE_DIRS}
)

# Subdirectories
add_subdirectory(src)
add_subdirectory(tests)

# Executable with explicit linking
add_executable(semantic_search src/main.cpp)
target_link_libraries(semantic_search PRIVATE 
    similarity_utils
    cpu_implementations
    gpu_implementations
    CUDA::cudart
    OpenMP::OpenMP_CXX
    ${FAISS_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    raft::compiled
)

# Debug output
message(STATUS "FAISS_LIBRARIES: ${FAISS_LIBRARIES}")
message(STATUS "BLAS_LIBRARIES: ${BLAS_LIBRARIES}")
message(STATUS "RAFT_INCLUDE_DIRS: ${RAFT_INCLUDE_DIRS}")