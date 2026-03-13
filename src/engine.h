#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Parâmetros de uma simulação — tudo C++ puro, sem pybind11
struct SimParams {
    std::string id;
    json        params;

    // entry/exit binários exclusivos desta sim (uint8: 0/1)
    // Buffers próprios — ponteiros estáveis durante o paralelo
    std::unordered_map<std::string, std::vector<uint8_t>> signal_array_bufs;

    // sl_price, tp_price, limit, trail, be → nome de coluna no fast_pool
    std::unordered_map<std::string, std::string> signal_refs;
};

struct EngineResult {
    std::vector<json> simulations;   // [sim_idx] → json array de trades
    std::vector<json> wfm_data;
};

class Engine {
public:
    static EngineResult execute(
        const std::string&                                    header,
        const std::unordered_map<std::string, const double*>& ohlc_arrays,
        size_t                                                n_bars,
        const std::vector<int64_t>&                           datetime_int,
        const std::unordered_map<std::string, const double*>& indicators_pool,
        const std::unordered_map<std::string, const uint8_t*>& shared_signal_arrays,
        const std::vector<SimParams>&                         sim_params,
        const json&                                           exec_settings
    );
};