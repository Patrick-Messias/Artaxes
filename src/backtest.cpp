
#include "Backtest.h"
#include <iostream>
#include <random>
#include <sstream>
#include <algorithm>
#include <iomanip> // Para formatar logs

using json = nlohmann::json;

double calculate_lot_size(double price_entry, bool is_long_trade);

static std::string generate_id() {
    static std::mt19937_64 rng{std::random_device{}()};
    std::stringstream ss;
    ss << std::hex << rng();
    return ss.str();
}

std::vector<Trade> Backtest::run(const std::string& dataset_key,
                                 const json& dataset,
                                 const json& meta) {
    std::vector<Trade> trades;
    std::vector<Trade> active_trades;
    int log_count = 0; // Contador para não inundar o console
    int log_size = 50;

    try {
        if (!dataset.contains("data")) return {};
        const auto& d = dataset.at("data");

        auto num_pos_json = meta.value("strat_num_pos", json::array({1, 1}));
        int max_long = num_pos_json[0].get<int>();
        int max_short = num_pos_json[1].get<int>();
        bool hedge_enabled = meta.value("hedge", false);

        auto datetime = d.at("datetime").get<std::vector<std::string>>();
        auto open = d.at("open").get<std::vector<double>>();
        auto high = d.at("high").get<std::vector<double>>();
        auto low = d.at("low").get<std::vector<double>>();
        auto close = d.at("close").get<std::vector<double>>();

        auto get_vec = [&](const std::string& key) {
            return d.contains(key) ? d.at(key).get<std::vector<int>>() : std::vector<int>(datetime.size(), 0);
        };
        auto get_double_vec = [&](const std::string& key) {
            return d.contains(key) ? d.at(key).get<std::vector<double>>() : std::vector<double>(datetime.size(), 0.0);
        };

        auto entry_long = get_vec("entry_long");
        auto entry_short = get_vec("entry_short");
        auto exit_tf_long = get_vec("exit_tf_long");
        auto exit_tf_short = get_vec("exit_tf_short");
        auto sl_long_v = get_double_vec("exit_sl_long");
        auto tp_long_v = get_double_vec("exit_tp_long");
        auto sl_short_v = get_double_vec("exit_sl_short");
        auto tp_short_v = get_double_vec("exit_tp_short");
        
        for (size_t i = 1; i < datetime.size(); ++i){
            double curr_open = open[i];
            int prev = (i > 0) ? i - 1 : 0; //size_t prev = i-1;

            // --- 1. EXIT BY SIGNAL (TF) ---
            for (auto it = active_trades.begin(); it != active_trades.end(); ){
                if (it->entry_datetime == datetime[i]) {
                        ++it; 
                        continue; // Pula este trade, ele acabou de abrir, não pode fechar agora.
                    }

                bool is_l = it->lot_size.value_or(0) > 0;
                
                // Se o seu objeto Trade não tem entry_index, o lucro zero é quase certo se os sinais coincidirem.
                if((is_l && exit_tf_long[prev]) || (!is_l && exit_tf_short[prev])) {


                    it->exit_price = curr_open;
                    it->exit_datetime = datetime[i];
                    it->exit_reason = "tf_exit";
                    it->status = "closed";

                    double entry_p = it->entry_price.value_or(0.0);
                    if (entry_p > 0.0) {
                        double pnl_raw = is_l ? (curr_open - entry_p) : (entry_p - curr_open);
                        it->profit = (pnl_raw / entry_p) * 100.0;
                    }

                    // LOG DOS PRIMEIROS TRADES
                    if (log_count < log_size) {
                        // 1. Extraímos os valores para variáveis locais double puras
                        // O cast (double) garante que o compilador trate como número, não como optional
                        double p_entry  = it->entry_price ? (double)*it->entry_price : 0.0;
                        double p_exit   = it->exit_price  ? (double)*it->exit_price  : 0.0;
                        double p_profit = it->profit      ? (double)*it->profit      : 0.0;

                        // 2. Agora imprimimos apenas as variáveis locais
                        int side = is_l ? (1) : (-1);
                        std::cout << "[C++ LOG] EXIT tf | Data: " << datetime[i];
                        std::cout << " | Entry: " << p_entry;
                        std::cout << " | Exit: " << p_exit;
                        std::cout << " | Side: "  << side;
                        std::cout << " | PnL: " << p_profit << "%" << std::endl;
                        
                        log_count++;
                    }

                    trades.push_back(*it);
                    it = active_trades.erase(it);
                } else { ++it; }
            }

            // --- 2. ENTRY ---
            int cur_l=0, cur_s=0;
            for(const auto& t: active_trades) (t.lot_size.value_or(0) > 0) ? cur_l++ : cur_s++;
            
            auto try_entry = [&](bool is_long_side) {
                Trade t;
                t.id = generate_id();
                t.asset = dataset_key;
                t.entry_datetime = datetime[i];
                t.entry_price = curr_open;
                t.status = "open";
                t.lot_size = calculate_lot_size(curr_open, is_long_side);

                double sl_val = is_long_side ? sl_long_v[prev] : sl_short_v[prev];
                double tp_val = is_long_side ? tp_long_v[prev] : tp_short_v[prev];

                if (sl_val > 0.0) {
                    t.stop_loss = is_long_side ? (curr_open - sl_val) : (curr_open + sl_val);
                } else {
                    t.stop_loss = std::nullopt;
                }

                if (tp_val > 0.0) {
                    t.take_profit = is_long_side ? (curr_open + tp_val) : (curr_open - tp_val);
                } else {
                    t.take_profit = std::nullopt;
                }
            
                if (log_count < log_size) {
                    int side = is_long_side ? (1) : (-1);
                    std::cout << "[C++ LOG] ENTRY | Data: " << datetime[i] << " | Price: " << curr_open << " | SL: " << sl_val << " | Side: " << side << std::endl;
                }
                active_trades.push_back(t);
            };

            if (entry_long[prev] && (hedge_enabled ? cur_l < max_long : (cur_l < max_long && cur_s == 0)))
                try_entry(true);
            if (entry_short[prev] && (hedge_enabled ? cur_s < max_short : (cur_s < max_short && cur_l == 0)))
                try_entry(false);
            
            // --- 3. EXIT INTRA-CANDLE (SL/TP) ---
            for (auto it = active_trades.begin(); it != active_trades.end(); ){
                bool hit = false;
                double exit_p = 0;
                std::string reason = "";
                bool is_l = it->lot_size.value_or(0) > 0;
                double sl = it->stop_loss.value_or(0.0);
                double tp = it->take_profit.value_or(0.0);

                if (is_l) {
                    if (close[i] > open[i]) {
                        if (sl > 0 && low[i] <= sl) { hit=true; exit_p=sl; reason="sl"; }
                        else if (tp > 0 && high[i] >= tp) { hit=true; exit_p=tp; reason="tp"; }
                    } else {
                        if (tp > 0 && high[i] >= tp) { hit=true; exit_p=tp; reason="tp"; }
                        else if (sl > 0 && low[i] <= sl) { hit=true; exit_p=sl; reason="sl"; }
                    }
                } else {
                    if (close[i] > open[i]) {
                        if (tp > 0 && low[i] <= tp) { hit=true; exit_p=tp; reason="tp"; }
                        else if (sl > 0 && high[i] >= sl) { hit=true; exit_p=sl; reason="sl"; }
                    } else {
                        if (sl > 0 && high[i] >= sl) { hit=true; exit_p=sl; reason="sl"; }
                        else if (tp > 0 && low[i] <= tp) { hit=true; exit_p=tp; reason="tp"; }
                    }
                }

                if (hit) {
                    it->exit_price = exit_p;
                    it->exit_datetime = datetime[i];
                    it->exit_reason = reason;
                    it->status = "closed";
                    double entry_p = it->entry_price.value_or(0.0);
                    if (entry_p > 0.0) {
                        double pnl_raw = is_l ? (exit_p - entry_p) : (entry_p - exit_p);
                        it->profit = (pnl_raw / entry_p) * 100.0;
                    }
                    
                    if (log_count < log_size) {
                        // 1. Extraímos os valores para variáveis locais double puras
                        // O cast (double) garante que o compilador trate como número, não como optional
                        double p_entry  = it->entry_price ? (double)*it->entry_price : 0.0;
                        double p_exit   = it->exit_price  ? (double)*it->exit_price  : 0.0;
                        double p_profit = it->profit      ? (double)*it->profit      : 0.0;

                        // 2. Agora imprimimos apenas as variáveis locais
                        int side = is_l ? (1) : (-1);
                        std::cout << "[C++ LOG] EXIT " << reason << " | Data: " << datetime[i];
                        std::cout << " | Entry: " << p_entry;
                        std::cout << " | Exit: " << p_exit;
                        std::cout << " | Side: " << side;
                        std::cout << " | PnL: " << p_profit << "%" << std::endl;
                        
                        log_count++;
                    }

                    trades.push_back(*it);
                    it = active_trades.erase(it);
                } else { ++it; }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[C++ Error]: " << e.what() << std::endl;
    }
    return trades;
}

double calculate_lot_size(double price_entry, bool is_long_trade) {
    return (is_long_trade) ? 1.0 : -1.0;
}









