cmake_minimum_required(VERSION 3.22)
project(mem_cuda CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_COMPILER "/usr/bin/clang++-12")
set(CMAKE_C_COMPILER "/usr/bin/clang-12")
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")

add_executable(mem_cuda main.cu)

set_target_properties(
        mem_cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)