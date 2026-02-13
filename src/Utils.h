#pragma once
#include <vector>
#include <nlohmann/json.hpp>
#include "Trade.h"

using json = nlohmann::json;

inline nlohmann::json trades_to_json(const std::vector<Trade>& trades) {
    nlohmann::json j = nlohmann::json::array();
    for (const auto& t : trades) {
        j.push_back({
            {"id", t.id},
            {"asset", t.asset},
            {"status", t.status},
            {"entry_price", t.entry_price},
            {"entry_datetime", t.entry_datetime},
            {"lot_size", t.lot_size},
            {"stop_loss", t.stop_loss ? json(*t.stop_loss) : json(nullptr)},
            {"take_profit", t.take_profit ? json(*t.take_profit) : json(nullptr)},
            {"exit_price", t.exit_price ? json(*t.exit_price) : json(nullptr)},
            {"exit_datetime", t.exit_datetime ? json(*t.exit_datetime) : json(nullptr)},
            {"exit_reason", t.exit_reason ? json(*t.exit_reason) : json(nullptr)},
            {"profit", t.profit ? json(*t.profit) : json(nullptr)},
            {"profit_r", t.profit_r ? json(*t.profit_r) : json(nullptr)},
            {"position_value", t.position_value ? json(*t.position_value) : json(nullptr)},
            {"mfe", t.mfe ? json(*t.mfe) : json(nullptr)},
            {"mae", t.mae ? json(*t.mae) : json(nullptr)},
            {"bars_held", t.bars_held ? json(*t.bars_held) : json(nullptr)}
        });
    }
    return j;
}



#include <fstream>

void export_to_csv(const std::string& filename, 
                   const std::vector<std::string>& datetime, 
                   const std::map<std::string, std::vector<double>>& data) {
    std::ofstream file(filename);
    
    // 1. Escrever o Cabeçalho
    file << "datetime";
    for (auto const& [col_name, _] : data) {
        file << "," << col_name;
    }
    file << "\n";

    // 2. Escrever os Dados
    size_t n_bars = datetime.size();
    for (size_t i = 0; i < n_bars; ++i) {
        file << datetime[i];
        for (auto const& [col_name, vec] : data) {
            if (i < vec.size()) {
                file << "," << std::fixed << std::setprecision(6) << vec[i];
            } else {
                file << ",NaN";
            }
        }
        file << "\n";
    }

    file.close();
}



#include <ctime>
#include <iomanip>
#include <sstream>

int get_day_of_week(const std::string& dt_str) {
    std::tm tm = {};
    std::istringstream ss(dt_str);
    ss >> std::get_time(&tm, "%Y-%m-%d"); // Ajuste o formato se necessário
    if (ss.fail()) return -1;
    
    std::mktime(&tm); // Normaliza e preenche o tm_wday
    return tm.tm_wday; // 0 = Domingo, 1 = Segunda, ..., 5 = Sexta, 6 = Sábado
}




