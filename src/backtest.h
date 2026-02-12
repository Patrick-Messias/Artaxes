#pragma once
#include <vector>
#include <string>
#include <map>
#include <nlohmann/json.hpp>
#include "Trade.h"

class Backtest {
public:
    static nlohmann::json run_simulation(const std::string& header, 
                                        const std::map<std::string, std::vector<double>>& data,
                                        const std::vector<std::string>& datetime,
                                        const nlohmann::json& sim,
                                        const nlohmann::json& exec_settings);
};
