#pragma once
#include <vector>
#include <nlohmann/json.hpp>
#include "Trade.h"
#include <fstream>
#include <string>

/**
 * Converte o vetor de objetos Trade para JSON, aplicando o filtro de resolução de PnL.
 * @param trades Vetor de trades finalizados.
 * @param resolution Resolução desejada: "daily", "weekly", ou "monthly".
 */
inline nlohmann::json trades_to_json(const std::vector<Trade>& trades, const std::string& resolution = "daily") {
    nlohmann::json j = nlohmann::json::array();

    for (const auto& t : trades) {
        // Vetores que serão enviados no JSON
        std::vector<double> final_pnl;
        std::vector<std::string> final_dates;

        if (resolution == "daily" || t.daily_datetime.empty()) { 
            final_pnl = t.daily_pnl;
            final_dates = t.daily_datetime;
        } else {
            // Lógica de Downsampling (Redução de Resolução)
            for (size_t k = 0; k < t.daily_datetime.size(); ++k) {
                bool is_last = (k == t.daily_datetime.size() - 1);
                bool include = false;

                if (is_last) {
                    include = true; // Sempre inclui o último registro do trade (fechamento)
                } else if (resolution == "monthly") {
                    // Inclui se o mês da data atual for diferente do próximo (YYYY-MM-DD)
                    // Substr(5,2) extrai o "MM"
                    if (t.daily_datetime[k].substr(5, 2) != t.daily_datetime[k+1].substr(5, 2)) {
                        include = true;
                    }
                } else if (resolution == "weekly") {
                    // Abordagem simples para Weekly: Se o mês mudou ou se é um salto de 5 registros (dias úteis)
                    // Como não temos um calendário complexo aqui, a virada de mês serve como checkpoint
                    if (t.daily_datetime[k].substr(5, 2) != t.daily_datetime[k+1].substr(5, 2) || k % 5 == 0) {
                        include = true;
                    }
                }

                if (include) {
                    final_pnl.push_back(t.daily_pnl[k]);
                    final_dates.push_back(t.daily_datetime[k]);
                }
            }
        }

        // Montagem do objeto JSON para cada trade
        nlohmann::json trade_json = {
            {"id", t.id},
            {"asset", t.asset},
            {"status", t.status},
            {"entry_price", t.entry_price},
            {"entry_datetime", t.entry_datetime},
            {"lot_size", t.lot_size},
            {"stop_loss", t.stop_loss ? nlohmann::json(*t.stop_loss) : nlohmann::json(nullptr)},
            {"take_profit", t.take_profit ? nlohmann::json(*t.take_profit) : nlohmann::json(nullptr)},
            {"exit_price", t.exit_price ? nlohmann::json(*t.exit_price) : nlohmann::json(nullptr)},
            {"exit_datetime", t.exit_datetime ? nlohmann::json(*t.exit_datetime) : nlohmann::json(nullptr)},
            {"exit_reason", t.exit_reason ? nlohmann::json(*t.exit_reason) : nlohmann::json(nullptr)},
            {"profit", t.profit ? nlohmann::json(*t.profit) : nlohmann::json(nullptr)},
            {"profit_r", t.profit_r ? nlohmann::json(*t.profit_r) : nlohmann::json(nullptr)},
            {"mfe", t.mfe ? nlohmann::json(*t.mfe) : nlohmann::json(nullptr)},
            {"mae", t.mae ? nlohmann::json(*t.mae) : nlohmann::json(nullptr)},
            {"bars_held", t.bars_held ? nlohmann::json(*t.bars_held) : nlohmann::json(nullptr)},
            
            // Campos obrigatórios para evitar KeyError no Python
            {"daily_pnl", final_pnl}, 
            {"daily_datetime", final_dates}
        };

        j.push_back(trade_json);
    }
    return j;
}



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




