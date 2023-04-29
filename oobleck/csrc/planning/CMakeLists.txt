cmake_minimum_required(VERSION 3.13)

project(cplanning)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

pybind11_add_module(cplanning gpu_stage_mapping.cpp)

# set(VERSION_INFO 0.1.0)
target_compile_definitions(cplanning PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_include_directories(cplanning
PRIVATE ${CMAKE_SOURCE_DIR}/modules/json/single_include
PRIVATE ${CMAKE_SOURCE_DIR}/modules/concurrentqueue
PRIVATE ${CMAKE_SOURCE_DIR}/modules/thread-pool)