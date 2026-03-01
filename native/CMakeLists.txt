cmake_minimum_required(VERSION 3.16)

project(trix_native_tools LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-O3)

# Apple Silicon tuning
if(APPLE)
  add_compile_options(-mcpu=native)
endif()

add_executable(trix_routebench
  src/main.cpp
  src/routing.cpp
  src/metrics.cpp
  src/jsonl.cpp
)

target_include_directories(trix_routebench PRIVATE src)

include(CTest)
if(BUILD_TESTING)
  add_executable(trix_native_metrics_tests
    tests/test_metrics.cpp
    src/metrics.cpp
  )
  target_include_directories(trix_native_metrics_tests PRIVATE src)
  add_test(NAME trix_native_metrics_tests COMMAND trix_native_metrics_tests)

  add_executable(trix_native_routing_tests
    tests/test_routing_invariants.cpp
    src/routing.cpp
    src/metrics.cpp
  )
  target_include_directories(trix_native_routing_tests PRIVATE src)
  add_test(NAME trix_native_routing_tests COMMAND trix_native_routing_tests)

  add_executable(trix_native_falsify_tests
    tests/test_falsify.cpp
    src/routing.cpp
    src/metrics.cpp
  )
  target_include_directories(trix_native_falsify_tests PRIVATE src)
  add_test(NAME trix_native_falsify_tests COMMAND trix_native_falsify_tests)
endif()
