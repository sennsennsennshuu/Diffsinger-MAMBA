cmake_minimum_required(VERSION 3.16)
project(SSMOptimizer VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(SSM_USE_SIMD "Enable SIMD optimizations (AVX/AVX2)" ON)
option(SSM_USE_OPENMP "Enable OpenMP parallelization" ON)
option(SSM_BUILD_TESTS "Build tests" ON)

# Compiler flags
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /fp:fast")
    if(SSM_USE_SIMD)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
    endif()
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")
    if(SSM_USE_SIMD)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    endif()
endif()

# OpenMP
if(SSM_USE_OPENMP)
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

# Source files
set(SSM_SOURCES
    src/ssm_optimizer.cpp
    src/selective_scan.cpp
    src/state_cache.cpp
    src/onnx_custom_op.cpp
)

# Header files
set(SSM_HEADERS
    ssm_optimizer.h
    src/selective_scan.h
    src/state_cache.h
    src/simd_utils.h
)

# Create shared library
add_library(SSMOptimizer SHARED ${SSM_SOURCES} ${SSM_HEADERS})

target_compile_definitions(SSMOptimizer PRIVATE SSM_OPTIMIZER_EXPORTS)

if(SSM_USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(SSMOptimizer OpenMP::OpenMP_CXX)
endif()

# ONNX Runtime
# Note: User needs to set ONNXRUNTIME_ROOT to their ONNX Runtime installation
if(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
else()
    set(ONNXRUNTIME_ROOT "C:/Program Files/onnxruntime")
endif()

target_include_directories(SSMOptimizer PRIVATE
    ${ONNXRUNTIME_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_directories(SSMOptimizer PRIVATE
    ${ONNXRUNTIME_ROOT}/lib
)

target_link_libraries(SSMOptimizer
    onnxruntime
)

# Install
install(TARGETS SSMOptimizer
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES ssm_optimizer.h DESTINATION include)

# Tests
if(SSM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
