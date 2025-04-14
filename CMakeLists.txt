cmake_minimum_required(VERSION 3.14)
project(SemanticSearch LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Find required packages
find_package(OpenMP REQUIRED)

# Add include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)

# Create executable
add_executable(semantic_search src/main.cpp)
target_link_libraries(semantic_search PRIVATE 
    similarity_utils
    cpu_implementations
    gpu_implementations
    ${CUDA_LIBRARIES}
    OpenMP::OpenMP_CXX
)