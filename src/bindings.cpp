#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <nlohmann/json.hpp>
#include <msgpack.hpp>
#include "engine.h"

namespace py = pybind11;

// Converte msgpack::object -> nlohmann::json recursivamente
nlohmann::json msgpack_to_json(const msgpack::object& obj) {
    switch (obj.type) {
        case msgpack::type::MAP: {
            nlohmann::json result = nlohmann::json::object();
            for (uint32_t i = 0; i < obj.via.map.size; ++i) {
                std::string key = obj.via.map.ptr[i].key.as<std::string>();
                result[key] = msgpack_to_json(obj.via.map.ptr[i].val);
            }
            return result;
        }
        case msgpack::type::ARRAY: {
            nlohmann::json result = nlohmann::json::array();
            for (uint32_t i = 0; i < obj.via.array.size; ++i)
                result.push_back(msgpack_to_json(obj.via.array.ptr[i]));
            return result;
        }
        case msgpack::type::STR:              return obj.as<std::string>();
        case msgpack::type::FLOAT32:
        case msgpack::type::FLOAT64:          return obj.as<double>();
        case msgpack::type::POSITIVE_INTEGER: return obj.as<uint64_t>();
        case msgpack::type::NEGATIVE_INTEGER: return obj.as<int64_t>();
        case msgpack::type::BOOLEAN:          return obj.as<bool>();
        default:                              return nullptr;
    }
}

PYBIND11_MODULE(engine_cpp, m) {
    m.doc() = "ART Engine Core";

    // Binding original — mantido para fallback/debug
    m.def("run", [](const std::string& json_input) {
        Engine engine;
        return engine.run(json_input);
    }, "Executa backtest via JSON string");

    // Novo binding — recebe bytes msgpack
    m.def("run_msgpack", [](py::bytes data) {
        std::string raw = static_cast<std::string>(data);
        msgpack::object_handle oh = msgpack::unpack(raw.data(), raw.size());
        nlohmann::json payload = msgpack_to_json(oh.get());
        Engine engine;
        return engine.run_from_json(payload);
    }, "Executa backtest via MessagePack bytes");
}