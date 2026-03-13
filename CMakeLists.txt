cmake_minimum_required(VERSION 3.18)
project(backtest_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYTHON_HOME "C:/Program Files/Python311")
set(Python3_ROOT_DIR "${PYTHON_HOME}")
set(Python3_EXECUTABLE "${PYTHON_HOME}/python.exe")
set(Python3_INCLUDE_DIR "${PYTHON_HOME}/include")
set(Python3_LIBRARY "${PYTHON_HOME}/libs/python311.lib")

find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(TBB REQUIRED)

pybind11_add_module(engine_cpp
    src/bindings.cpp
    src/engine.cpp
    src/operation.cpp
    src/backtest.cpp
)

target_include_directories(engine_cpp PRIVATE src)

target_link_libraries(engine_cpp PRIVATE
    pybind11::module
    nlohmann_json::nlohmann_json
    TBB::tbb
)

if(MSVC)
    target_compile_options(engine_cpp PRIVATE /O2 /Oi /Ot /EHsc)
else()
    target_compile_options(engine_cpp PRIVATE -O3)
endif()

if(WIN32)
    set_target_properties(engine_cpp PROPERTIES PREFIX "" SUFFIX ".pyd")
endif()