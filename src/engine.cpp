#include "engine.h"
#include "operation.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>
#include <iomanip>

using json = nlohmann::json;

std::string Engine::run(const std::string& payload_json) {
    try {
        if (payload_json.empty()) return "[]";

        json payload = json::parse(payload_json);

        std::string header = payload["asset_header"];
        auto datetime = payload["data"]["datetime"].get<std::vector<std::string>>();
        auto sim_params = payload["simulations"];
        auto exec_settings = payload["execution_settings"];
        auto time_settings = payload["time_settings"];

        std::map<std::string, std::vector<double>> data_map;
        
        std::cout << "\n[C++ DEBUG] --- INSPECIONANDO PAYLOAD RECEBIDO ---" << std::endl;
        std::cout << " > Colunas detectadas no JSON: ";
        for (auto it = payload["data"].begin(); it != payload["data"].end(); ++it) {
            std::cout << "[" << it.key() << "] ";
        }
        std::cout << "\n" << std::endl;

        for (auto& el : payload["data"].items()) {
            std::string col_name = el.key();
            if (col_name == "datetime") continue;

            try {
                std::vector<double> values = el.value().get<std::vector<double>>();
                data_map[col_name] = values;

                // Print dos índices 50 a 60
                std::cout << " > Inspecionando coluna '" << col_name << "' (indices 50-60):" << std::endl;
                if (values.size() > 60) {
                    std::cout << "   Values: ";
                    for (int i = 50; i <= 60; ++i) {
                        std::cout << std::fixed << std::setprecision(5) << values[i] << (i == 60 ? "" : ", ");
                    }
                    std::cout << std::endl;
                } else {
                    std::cout << "   [AVISO] Coluna muito curta para amostra 50-60. Tamanho: " << values.size() << std::endl;
                }

            } catch (const std::exception& e) {
                std::cerr << " > ERROR na coluna '" << col_name << "': " << e.what() << std::endl;
            }
        }

        // Verificação específica para MA nas simulações
        std::cout << "\n[C++ DEBUG] Verificando 'indicator_data' na primeira simulacao:" << std::endl;
        if (!sim_params.empty() && sim_params[0].contains("indicator_data")) {
            for (auto& el : sim_params[0]["indicator_data"].items()) {
                std::cout << " > Indicador extra detectado: [" << el.key() << "]" << std::endl;
            }
        } else {
            std::cout << " > [AVISO] 'indicator_data' nao encontrado ou vazio nas simulacoes." << std::endl;
        }

        std::cout << "\n[C++ DEBUG] Payload processado. Chamando Operation::run...\n" << std::endl;
        
        json results = Operation::run(header, data_map, datetime, sim_params, exec_settings, time_settings);
        return results.dump();

    } catch (const std::exception& e) {
        std::cerr << "[C++ Engine Error]: " << e.what() << std::endl;
        return "[]";
    }
}