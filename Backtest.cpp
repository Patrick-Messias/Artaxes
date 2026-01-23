#include "Backtest.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

Backtest::Backtest(const std::vector<SignalData>& data)
    : data_(data) {}

void Backtest::run() {
    // Placeholder: aqui entra a lógica real depois
}

std::string Backtest::get_results_json() const {
    json result;
    result["status"] = "ok";
    result["message"] = "Backtest executed successfully";
    result["assets"] = data_.size();
    return result.dump();
}
