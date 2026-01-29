#include "Backtest.h"
#include <iostream>
#include <random>
#include <sstream>
#include <algorithm>

using json = nlohmann::json;

// Gerador de ID único
static std::string generate_id() {
    static std::mt19937_64 rng{std::random_device{}()};
    std::stringstream ss;
    ss << std::hex << rng();
    return ss.str();
}

static bool is_friday(const std::string& datetime) {
    return false; // Implementar via <chrono> se necessário
}

std::vector<Trade> Backtest::run(const std::string& dataset_key,
                                 const json& dataset,
                                 const json& meta) {
    std::vector<Trade> trades;

    try {
        // 1. Acesso seguro às seções
        if (!dataset.contains("data")) return {};
        const auto& df_json = dataset.at("data");
        const auto& time_settings = dataset.at("time_settings");

        // 2. Extração de vetores com verificação de existência
        // Se a coluna não existir, inicializa vetor vazio para não travar
        auto get_v_double = [&](const std::string& key) {
            return df_json.contains(key) ? df_json.at(key).get<std::vector<double>>() : std::vector<double>();
        };
        auto get_v_str = [&](const std::string& key) {
            return df_json.contains(key) ? df_json.at(key).get<std::vector<std::string>>() : std::vector<std::string>();
        };
        auto get_v_int = [&](const std::string& key) {
            return df_json.contains(key) ? df_json.at(key).get<std::vector<int>>() : std::vector<int>();
        };

        std::vector<double> open = get_v_double("open");
        std::vector<double> high = get_v_double("high");
        std::vector<double> low = get_v_double("low");
        std::vector<double> close = get_v_double("close");
        std::vector<std::string> datetime = get_v_str("datetime");

        std::vector<int> entry_long = get_v_int("entry_long");
        std::vector<int> entry_short = get_v_int("entry_short");
        std::vector<int> exit_tf_long = get_v_int("exit_tf_long");
        std::vector<int> exit_tf_short = get_v_int("exit_tf_short");
        std::vector<int> be_pos_long = get_v_int("be_pos_long");
        std::vector<int> be_pos_short = get_v_int("be_pos_short");

        size_t n = open.size();
        
        // 3. Validação de Alinhamento (Evita Access Violation)
        if (n < 2) return {};
        if (entry_long.size() < n || entry_short.size() < n || datetime.size() < n) {
            std::cerr << "[C++] ERRO: Vetores desalinhados para " << dataset_key << std::endl;
            return {};
        }

        bool day_trade = time_settings.value("day_trade", false);
        bool in_position = false;
        bool is_long = false;
        Trade current_trade;
        double entry_price_val = 0.0;
        double stop_loss = 0.0;
        double take_profit = 0.0;
        double trailing_sl = 0.0;

        // 4. Loop de Backtest
        for (size_t i = 1; i < n; ++i) {
            if (!in_position) {
                // Sinais de Entrada
                if (entry_long[i] == 1) {
                    in_position = true; is_long = true;
                    entry_price_val = open[i];
                    stop_loss = low[i];
                    take_profit = high[i];
                    trailing_sl = stop_loss;
                    current_trade = { generate_id(), dataset_key, "open", "long", entry_price_val, datetime[i], 1.0 };
                    current_trade.stop_loss = stop_loss;
                    current_trade.take_profit = take_profit;
                } else if (entry_short[i] == 1) {
                    in_position = true; is_long = false;
                    entry_price_val = open[i];
                    stop_loss = high[i];
                    take_profit = low[i];
                    trailing_sl = stop_loss;
                    current_trade = { generate_id(), dataset_key, "open", "short", entry_price_val, datetime[i], 1.0 };
                    current_trade.stop_loss = stop_loss;
                    current_trade.take_profit = take_profit;
                }
            } else {
                // Gestão de Saída
                bool exit = false;
                std::string reason;
                double exit_price = open[i];

                // Trailing & Stop/TP básico
                if (is_long) {
                    trailing_sl = std::max(trailing_sl, low[i]);
                    if (low[i] <= stop_loss) { exit = true; reason = "stop_loss"; exit_price = stop_loss; }
                    else if (high[i] >= take_profit) { exit = true; reason = "take_profit"; exit_price = take_profit; }
                    else if (low[i] <= trailing_sl) { exit = true; reason = "trailing_stop"; exit_price = trailing_sl; }
                    else if (!exit_tf_long.empty() && exit_tf_long[i]) { exit = true; reason = "tf_exit"; }
                } else {
                    trailing_sl = std::min(trailing_sl, high[i]);
                    if (high[i] >= stop_loss) { exit = true; reason = "stop_loss"; exit_price = stop_loss; }
                    else if (low[i] <= take_profit) { exit = true; reason = "take_profit"; exit_price = take_profit; }
                    else if (high[i] >= trailing_sl) { exit = true; reason = "trailing_stop"; exit_price = trailing_sl; }
                    else if (!exit_tf_short.empty() && exit_tf_short[i]) { exit = true; reason = "tf_exit"; }
                }

                if (exit) {
                    double pnl = is_long ? (exit_price - entry_price_val) : (entry_price_val - exit_price);
                    current_trade.status = "closed";
                    current_trade.exit_price = exit_price;
                    current_trade.exit_datetime = datetime[i];
                    current_trade.exit_reason = reason;
                    current_trade.profit = (pnl / entry_price_val) * 100.0;
                    current_trade.profit_r = pnl;
                    trades.push_back(current_trade);
                    in_position = false;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[C++] Exception in Backtest::run: " << e.what() << std::endl;
    }

    return trades;
}