#pragma once
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include "Trade.h"
#include "Utils.h"

struct SimulationOutput {
    nlohmann::json trades_json;
    std::vector<DailyResult> daily_vec;
};

class Backtest {
public:
    using ColPtr = const std::vector<double>*;
    using SimView = std::unordered_map<std::string, ColPtr>;
    
    static SimulationOutput run_simulation(
        const std::string& header, 
        const std::map<std::string, std::vector<double>>& base_data,
        const SimView& sim_view,
        const std::vector<std::string>& datetime,
        const nlohmann::json& sim,
        const nlohmann::json& exec_settings,
        int ps_id // To identify which param_set this is, corresponds to the index in param_sets in Python
        );
};
