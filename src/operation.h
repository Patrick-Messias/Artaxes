#ifndef OPERATION_H
#define OPERATION_H

#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include <map> // Essencial para o std::map
#include "backtest.h"

using json = nlohmann::json;

class Operation {
public: // Static so that Engine will call without initiating a class
    static json run(const std::string& header, 
                    const std::map<std::string, std::vector<double>>& data,
                    const std::vector<std::string>& datetime,
                    const nlohmann::json& sim_params,
                    const nlohmann::json& exec_settings,
                    const nlohmann::json& shared_inds,
                    const nlohmann::json& payload_shared_sigs
                    );
};

#endif