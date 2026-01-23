#include "Engine.h"
#include "Backtest.h"
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ===============================
// Utilitário: converte objeto Pandas JSON em vetor
// ===============================
template<typename T>
std::vector<T> extractColumn(const json& colJson) {
    std::vector<T> result;
    for (auto& [idx, val] : colJson.items()) {
        if (val.is_null()) {
            result.push_back(T());
        } else {
            result.push_back(val.get<T>());
        }
    }
    return result;
}

// ===============================
// Parser principal
// ===============================
std::vector<SignalData> parseJsonToSignals(const std::string& jsonStr) {
    std::vector<SignalData> signals;
    json j = json::parse(jsonStr);

    for (auto& [key, value] : j.items()) {

        if (key == "pre_backtest_signal_is_position" ||
            key == "date_start" ||
            key == "date_end" ||
            key == "operation") {
            continue;
        }

        if (!value.is_object()) continue;

        SignalData data;
        data.key = key;

        if (value.contains("open"))     data.open = extractColumn<double>(value["open"]);
        if (value.contains("high"))     data.high = extractColumn<double>(value["high"]);
        if (value.contains("low"))      data.low = extractColumn<double>(value["low"]);
        if (value.contains("close"))    data.close = extractColumn<double>(value["close"]);
        if (value.contains("datetime")) data.datetime = extractColumn<std::string>(value["datetime"]);

        if (value.contains("entry_long")) {
            auto col = value["entry_long"];
            for (auto& [idx, val] : col.items()) {
                data.signals.push_back(val.is_null() ? 0 : (val.get<bool>() ? 1 : 0));
            }
        }

        signals.push_back(data);
    }

    return signals;
}

// ===============================
// Função exportada para Python
// ===============================
std::string run_backtest_from_json(const std::string& json_input) {
    std::cout << "Received JSON from Python:\n" << json_input << std::endl;

    auto signals = parseJsonToSignals(json_input);

    Backtest bt(signals);
    bt.run();

    return bt.get_results_json();
}
