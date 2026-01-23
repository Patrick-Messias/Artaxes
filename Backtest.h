#pragma once
#include <string>
#include <vector>

// ===============================
// Struct para armazenar um backtest unitário
// ===============================
struct SignalData {
    std::string key;
    std::vector<double> open;
    std::vector<double> high;
    std::vector<double> low;
    std::vector<double> close;
    std::vector<int> signals;
    std::vector<std::string> datetime;
};

// ===============================
// Classe principal de backtest
// ===============================
class Backtest {
public:
    Backtest(const std::vector<SignalData>& data);

    void run();
    std::string get_results_json() const;

private:
    std::vector<SignalData> data_;
};
