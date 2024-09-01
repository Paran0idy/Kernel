cmake_minimum_required(VERSION 3.20)
project(sgemm)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_language(CUDA) # Enable CUDA support, set this to your CUDA standard if needed.
set(CMAKE_CUDA_ARCHITECTURES 86)

set(INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include)
set(LIB_DIRS ${PROJECT_SOURCE_DIR}/lib)

include_directories(${INCLUDE_DIRS})

option(sgemm "Build sgemm executable" ON)
option(softmax "Build softmax executable" ON)
option(transpose "Build transpose executable" ON)

if (sgemm)
    add_executable(sgemm ${LIB_DIRS}/sgemm.cu)
    target_link_libraries(sgemm cublas)
endif ()

if (softmax)
    add_executable(softmax ${LIB_DIRS}/softmax.cu)
    target_link_libraries(softmax cublas)
endif ()

if (transpose)
    add_executable(transpose ${LIB_DIRS}/transpose.cu)
    target_link_libraries(transpose cublas)
endif ()