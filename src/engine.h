#ifndef ENGINE_H
#define ENGINE_H

#include <string>
#include <nlohmann/json.hpp>

class Engine {
public:
    // O retorno deve ser std::string para bater com o .cpp e com o Pybind11
    std::string run(const std::string& payload_json);
    std::string run_from_json(const nlohmann::json& payload);
};

#endif