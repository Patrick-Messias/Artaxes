cmake_minimum_required(VERSION 3.18)
project(backtest_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Forçar caminhos antes de qualquer find_package
set(PYTHON_HOME "C:/Program Files/Python311")
set(Python3_ROOT_DIR "${PYTHON_HOME}")
set(Python3_EXECUTABLE "${PYTHON_HOME}/python.exe")
set(Python3_INCLUDE_DIR "${PYTHON_HOME}/include")
set(Python3_LIBRARY "${PYTHON_HOME}/libs/python311.lib")

# 2. Impedir que o vcpkg redirecione a busca do Python
set(Python3_FIND_STRATEGY LOCATION)
set(Python3_FIND_REGISTRY NEVER)

# 3. Agora busca com os caminhos já pré-configurados
find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)
find_package(nlohmann_json REQUIRED)

# 4. Definição do módulo (certifique-se de que os nomes dos arquivos batem com sua pasta src)
pybind11_add_module(engine_cpp
    src/backtest.cpp
    src/engine.cpp
    src/bindings.cpp
)

target_link_libraries(engine_cpp PRIVATE 
    nlohmann_json::nlohmann_json
    Python3::Module
)

# 🪟 Garante que no Windows o arquivo final seja .pyd (ex: engine_cpp.pyd)
if (WIN32)
    set_target_properties(engine_cpp PROPERTIES
        PREFIX ""
        SUFFIX ".pyd"
    )
endif()