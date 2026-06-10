cmake_minimum_required(VERSION 3.24)

# Pick the visible GPU by default. Users can still override this, for example:
# cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to build for")
endif()

project(cuda_kernel_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_compile_options(
  $<$<COMPILE_LANGUAGE:CXX>:-Wall>
  $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
  $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_subdirectory(lessons/01_vector_add)
add_subdirectory(lessons/02_reduction)

# Add future lessons as they become executable, for example:
# add_subdirectory(lessons/03_transpose)
