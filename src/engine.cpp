#include "Engine.h"
#include "Backtest.h"
#include <nlohmann/json.hpp>
#include <execution>   
#include <mutex>
#include <iostream>
#include <vector>

using json = nlohmann::json;

std::vector<Trade> Engine::run(const std::string& payload_json) {
    std::vector<Trade> all_trades;
    std::mutex trades_mutex;

    std::cout << "\n[C++ Debug] --- Iniciando Chamada Engine::run ---" << std::endl;

    try {
        // 1. Parsing do JSON
        if (payload_json.empty()) {
            std::cerr << "[C++ Debug] Erro: String JSON recebida está vazia!" << std::endl;
            return {};
        }
        
        json payload = json::parse(payload_json);

        // 2. Diagnóstico de Chaves (Aqui vamos descobrir por que o erro 'datasets' ocorre)
        std::cout << "[C++ Debug] Chaves encontradas no nível principal: ";
        for (auto& el : payload.items()) {
            std::cout << "[" << el.key() << "] ";
        }
        std::cout << std::endl;

        // 3. Verificação de existência com fallback
        if (!payload.contains("datasets")) {
            std::cerr << "[C++ Debug] ERRO FATAL: Chave 'datasets' não existe no payload enviado pelo Python." << std::endl;
            return {}; // Retorna lista vazia para o Python não receber None
        }

        const auto& datasets = payload.at("datasets");
        const auto& meta = payload.contains("meta") ? payload.at("meta") : json::object();

        std::cout << "[C++ Debug] Numero de datasets (ativos) encontrados: " << datasets.size() << std::endl;

        // 4. Preparação e Processamento Sequencial (Para logs claros)
        std::vector<std::pair<std::string, json>> items;
        for (auto it = datasets.begin(); it != datasets.end(); ++it) {
            items.emplace_back(it.key(), it.value());
        }

        std::for_each(std::execution::seq, items.begin(), items.end(),
            [&](const auto& item) {
                const std::string& asset_name = item.first;
                const json& data_content = item.second;

                try {
                    // Executa o backtest
                    auto trades = Backtest::run(asset_name, data_content, meta);
                    
                    std::lock_guard<std::mutex> lock(trades_mutex);
                    all_trades.insert(all_trades.end(), trades.begin(), trades.end());
                    
                    std::cout << "[C++ Debug] Ativo '" << asset_name << "' processado com sucesso. Trades: " << trades.size() << std::endl;
                } catch (const std::exception& inner_e) {
                    std::cerr << "[C++ Debug] Erro no ativo '" << asset_name << "': " << inner_e.what() << std::endl;
                }
            }
        );

    } catch (const json::parse_error& e) {
        std::cerr << "[C++ Debug] Erro de Parse JSON: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[C++ Debug] Erro Geral C++: " << e.what() << std::endl;
    }

    std::cout << "[C++ Debug] --- Finalizando. Total de Trades: " << all_trades.size() << " ---\n" << std::endl;

    // Garante que o Python receba uma lista, mesmo que vazia, evitando NoneType Error
    return all_trades;
}