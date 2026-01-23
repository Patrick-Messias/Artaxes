#include <nlohmann/json.hpp>
using json = nlohmann::json;

json run_engine(const json& data) {
    double capital = data["capital"];
    auto prices = data["prices"];

    double final_value = capital;
    for (auto& p : prices) {
        final_value *= (1.0 + p.get<double>());
    }

    json result;
    result["final_value"] = final_value;
    result["num_trades"] = prices.size();
    return result;
}



/*
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>  // Biblioteca para JSON
#include <pybind11/pybind11.h>  // Para integração com Python

using json = nlohmann::json;
namespace py = pybind11;

// Struct para representar dados de sinais (exemplo básico)
struct SignalData {
    std::string key;  // Ex: "model_strat_asset_param"
    std::vector<double> open;
    std::vector<double> high;
    std::vector<double> low;
    std::vector<double> close;
    std::vector<int> signals;  // Sinais como inteiros (0/1)
    // Adicionar mais campos conforme necessário (ex: datetime, indicadores)
};

// Função para desserializar JSON e criar vector de SignalData
std::vector<SignalData> parseJsonToSignals(const std::string& jsonStr) {
    std::vector<SignalData> signals;
    json j = json::parse(jsonStr);

    for (auto& [key, value] : j.items()) {
        if (key == "pre_backtest_signal_is_position" || key == "date_start" || key == "date_end" || key == "operation") {
            // Pular variáveis globais por enquanto
            continue;
        }
        // Assumir que value é um DataFrame serializado em JSON
        // Para simplicidade, parse como map e extrair colunas
        // Nota: Pandas to_json() produz JSON de dict, ajustar conforme necessário
        SignalData data;
        data.key = key;
        // Exemplo: assumir colunas "open", "high", etc.
        // Aqui você implementaria a conversão real de JSON para vectors
        // Por exemplo:
        // if (value.contains("open")) data.open = value["open"].get<std::vector<double>>();
        // Similar para outros campos
        signals.push_back(data);
    }
    return signals;
}

// Função de backtest (deixar em branco por enquanto)
void runBacktest(const std::vector<SignalData>& signals) {
    // TODO: Implementar lógica de backtest
    // Ex: loop sobre sinais, calcular PnL, etc.
    std::cout << "Backtest executado (placeholder)" << std::endl;
}

// Função main para teste
int main() {
    // Exemplo de JSON (substitua por leitura de arquivo ou entrada)
    std::string sampleJson = R"(
    {
        "model1_strat1_asset1_param1": {"open": [1.0, 2.0], "high": [1.1, 2.1], "low": [0.9, 1.9], "close": [1.05, 2.05], "signals": [0, 1]},
        "pre_backtest_signal_is_position": true,
        "date_start": "2023-01-01",
        "date_end": "2023-12-31",
        "operation": {"type": "backtest"}
    }
    )";

    auto signals = parseJsonToSignals(sampleJson);
    runBacktest(signals);

    std::cout << "Número de conjuntos de sinais processados: " << signals.size() << std::endl;

    // Iterar sobre os sinais e mostrar key e amostra de dados
    for (const auto& signal : signals) {
        std::cout << "Key: " << signal.key << std::endl;
        std::cout << "  Open (amostra): ";
        for (size_t i = 0; i < std::min(signal.open.size(), size_t(3)); ++i) {
            std::cout << signal.open[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "  Signals (amostra): ";
        for (size_t i = 0; i < std::min(signal.signals.size(), size_t(3)); ++i) {
            std::cout << signal.signals[i] << " ";
        }
        std::cout << std::endl;
        // Adicionar amostras para high, low, close se necessário
    }

    return 0;
}

// Módulo pybind11 para integração com Python
PYBIND11_MODULE(engine, m) {
    m.def("run_backtest_from_json", [](const std::string& jsonStr) -> std::string {
        try {
            auto signals = parseJsonToSignals(jsonStr);
            runBacktest(signals);
            // Retornar resultados (placeholder; implementar retorno real de métricas)
            return "Backtest completed successfully. Processed " + std::to_string(signals.size()) + " signal sets.";
        } catch (const std::exception& e) {
            return "Error: " + std::string(e.what());
        }
    });
}
*/


























