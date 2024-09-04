cmake_minimum_required(VERSION 3.16)
project(Kernel)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)

set(INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/3rd/cutlass/include)

set(LIB_DIRS ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find all .cu
file(GLOB_RECURSE CC_FILES ${LIB_DIRS}/*.cu)

foreach(CU_FILE ${CC_FILES})
    get_filename_component(EXE_NAME ${CU_FILE} NAME_WE)
    add_executable(${EXE_NAME} ${CU_FILE})
    set_target_properties(${EXE_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_compile_options(${EXE_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -std=c++17 -O2 -arch=sm_80>
    )
    target_include_directories(${EXE_NAME} PRIVATE ${INCLUDE_DIRS})
endforeach()