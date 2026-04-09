#pragma once
#include <vector>
#include <nlohmann/json.hpp>
#include "Trade.h"
#include <fstream>
#include <string>


inline void export_to_csv(const std::string& filename, 
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

inline int get_day_of_week(const std::string& dt_str) {
    std::tm tm = {};
    std::istringstream ss(dt_str);
    ss >> std::get_time(&tm, "%Y-%m-%d"); // Ajuste o formato se necessário
    if (ss.fail()) return -1;
    
    std::mktime(&tm); // Normaliza e preenche o tm_wday
    return tm.tm_wday; // 0 = Domingo, 1 = Segunda, ..., 5 = Sexta, 6 = Sábado
}



inline int extract_minutes(const std::string& dt) {
    // Espera formato: "2019-04-11 12:50:00"
    // Posição das horas: 11 e 12
    // Posição dos minutos: 14 e 15
    
    if (dt.length() < 16) return -1; // String inválida ou sem horário

    try {
        // Converte os caracteres diretamente para inteiros (ASCII '0' é 48)
        int hh = (dt[11] - '0') * 10 + (dt[12] - '0');
        int mm = (dt[14] - '0') * 10 + (dt[15] - '0');
        
        return (hh * 60) + mm;
    } catch (...) {
        return -1;
    }
}


struct DailyResult {
    long long ts; // Format int 'YYYYMMDDHHMMSS', ex "2023-10-25 14:30:00" -> 20231025143000
    double pnl; // Raw pct_change returns from asset
    double lot_size; 
    double mae;
    double mfe;
    std::string trade_id; // Instead of full str with params just use ids to reduce size and translate later in py
};

inline void to_json(nlohmann::json& j, const DailyResult& res) {
    j = nlohmann::json{{"ts", res.ts}, {"pnl", res.pnl}, {"lot_size", res.lot_size}, {"mae", res.mae}, {"mfe", res.mfe}, {"id", res.trade_id}};
}

inline void from_json(const nlohmann::json& j, DailyResult& res) {
    j.at("ts").get_to(res.ts);
    j.at("pnl").get_to(res.pnl);
    j.at("lot_size").get_to(res.lot_size);
    j.at("mae").get_to(res.mae);
    j.at("mfe").get_to(res.mfe);
    j.at("trade_id").get_to(res.trade_id);
}

inline long long format_datetime_to_int_from_parts(int bar_date, int bar_time) {
    return (long long)bar_date * 1000000LL + bar_time;
}

#include <string>
#include <algorithm>

inline long long format_datetime_to_int(std::string dt_str) {
    // Remove '-', ':', e ' ' da string "2023-10-25 14:30:00"
    // Resultará em "20231025143000"
    dt_str.erase(std::remove(dt_str.begin(), dt_str.end(), '-'), dt_str.end());
    dt_str.erase(std::remove(dt_str.begin(), dt_str.end(), ':'), dt_str.end());
    dt_str.erase(std::remove(dt_str.begin(), dt_str.end(), ' '), dt_str.end());
    
    try {
        return std::stoll(dt_str);
    } catch (...) {
        return 0; // Fallback para data inválida
    }
}