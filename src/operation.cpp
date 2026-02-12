#include "operation.h"
#include "backtest.h"
#include <execution>
#include <mutex>
#include <map>
#include <algorithm>
#include <vector>
#include <iostream>

json Operation::run(const std::string& header, 
                    const std::map<std::string, std::vector<double>>& data,
                    const std::vector<std::string>& datetime,
                    const nlohmann::json& sim_params,
                    const nlohmann::json& exec_settings) {

    // Lista para armazenar os resultados de todas as simulações
    std::vector<json> all_somulations_results;
    std::mutex mtx;

    // 1. Convertemos o nlohmann::json (sim_params) em um std::vector<json>
    //isso resolve o erro de "no instance of overloaded function std::for_each matches the argument list"
    //pois garante iteradores compatíveis com std::execution::par
    std::vector<json> simulations;
    if (sim_params.is_array()) {
        for (const auto& s : sim_params) {
            simulations.push_back(s);
        }
    }

    // 2. Processamento paralelo por simulação (Cada Param Set em uma Thread)
    // Usamos o cabeçalho <execution> para rodar em múltiplos núcleos do processador
    std::for_each(std::execution::par, simulations.begin(), simulations.end(), [&](const json& sim) {
        // 1. Criamos uma CÓPIA local dos dados OHLC para esta thread
        std::map<std::string, std::vector<double>> local_data = data; 

        // 2. FUNDAMENTAL: Injetar os indicadores específicos desta simulação
        if (sim.contains("indicator_data") && sim["indicator_data"].is_object()) {
            for (auto& [key, val] : sim["indicator_data"].items()) {
                try {
                    local_data[key] = val.get<std::vector<double>>();
                    
                    // --- PRINT DE VALIDAÇÃO DE INJEÇÃO ---
                    std::cout << "[C++ Sim ID: " << sim["id"].get<std::string>().substr(0,8) 
                            << "] Indicador '" << key << "' injetado. "
                            << "Primeiro valor: " << local_data[key][0] 
                            << " | Ultimo: " << local_data[key].back() << std::endl;

                } catch (const std::exception& e) {
                    std::cerr << "[C++ Error] Falha ao injetar " << key << ": " << e.what() << std::endl;
                }
            }
        }

        // 3. Agora passamos o local_data (OHLC + INDICADORES)
        json trades = Backtest::run_simulation(header, local_data, datetime, sim, exec_settings);

        std::lock_guard<std::mutex> lock(mtx);
        all_somulations_results.push_back(trades);
    });

    // 3. Retorna o array de resultados (que o engine.cpp irá converter em string JSON)
    return all_somulations_results;
}