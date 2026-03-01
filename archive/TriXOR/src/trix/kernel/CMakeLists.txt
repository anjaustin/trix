cmake_minimum_required(VERSION 3.16)
project(trix LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Detect architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    message(STATUS "Building for ARM64 with NEON")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=armv8-a+simd")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    message(STATUS "Building for x86_64 (scalar fallback)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
else()
    message(STATUS "Building for ${CMAKE_SYSTEM_PROCESSOR}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# Build shared library
add_library(trix SHARED trix.cpp)

# Position independent code for shared library
set_target_properties(trix PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
    PUBLIC_HEADER trix.h
)

# Install targets
install(TARGETS trix
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)
