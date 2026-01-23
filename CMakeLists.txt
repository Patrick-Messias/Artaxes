cmake_minimum_required(VERSION 3.18)
project(engine_cpp)

set(CMAKE_CXX_STANDARD 20)

find_package(pybind11 REQUIRED)
find_package(nlohmann_json REQUIRED)

pybind11_add_module(engine_cpp bindings.cpp Engine.cpp)
target_link_libraries(engine_cpp PRIVATE nlohmann_json::nlohmann_json)
