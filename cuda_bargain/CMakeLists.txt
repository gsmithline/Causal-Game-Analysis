cmake_minimum_required(VERSION 3.18)
project(cuda_bargain LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

# CUDA architectures for various GPUs
set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")

# Source files
set(CUDA_SOURCES
    src/bargain_game.cu
)

set(CPP_SOURCES
    src/python_bindings.cpp
)

# Create the Python module
pybind11_add_module(cuda_bargain_core ${CUDA_SOURCES} ${CPP_SOURCES})

# Include directories
target_include_directories(cuda_bargain_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(cuda_bargain_core PRIVATE
    ${TORCH_LIBRARIES}
)

# Compiler flags
target_compile_options(cuda_bargain_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        --expt-relaxed-constexpr
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
    >
)

# Set output directory
set_target_properties(cuda_bargain_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python
)

# Install target
install(TARGETS cuda_bargain_core
    LIBRARY DESTINATION cuda_bargain
)
