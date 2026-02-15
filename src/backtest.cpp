
#include "Backtest.h"
#include "Trade.h"
#include "Utils.h"
#include <iostream>
#include <random>
#include <sstream>
#include <algorithm>
#include <iomanip> // Para formatar logs

using json = nlohmann::json;

double calculate_lot_size(double price_entry, bool is_long_trade);


// Auxiliary function to generate unique IDs for the trades
static std::string generate_id() {
    static std::mt19937_64 rng{std::random_device{}()};
    std::stringstream ss;
    ss << std::hex << rng();
    return ss.str();
}

double get_value_safe(const json& node, const json& params) {
    if (node.is_number()) return node.get<double>();
    
    if (node.is_object() && node.value("type", "") == "param") {
        std::string p_name = node.value("name", "");
        if (params.contains(p_name)) {
            // VERIFICAÇÃO CRUCIAL: Se for string (ex: "sma"), não tente converter para double
            if (params[p_name].is_number()) {
                return params[p_name].get<double>();
            }
        }
    }
    return 0.0;
}

double get_numeric_value(const json& node, const json& params) {
    if (node.is_number()) return node.get<double>();
    
    if (node.is_object()) {
        if (node.value("type", "") == "param") {
            return params.value(node.value("name", ""), 0.0);
        }
    }
    return 0.0;
}

// 1. Função que resolve qualquer conta (+, -, *, /) vinda do Python
double resolve_expression(const json& expr, const std::map<std::string, std::vector<double>>& data, const json& params, size_t i) {
    try {
        // Se for um número puro (ex: 0.001)
        if (expr.is_number()) return expr.get<double>();
        
        if (expr.is_object()) {
            std::string type = expr.value("type", "");

            // Caso 1: Referência de Parâmetro (sl_perc)
            if (type == "param") {
                std::string name = expr.value("name", "");
                auto& val = params[name];
                if (!val.is_number()) {
                    // LOG AQUI: Identifica se o parâmetro (ex: 'sma') está vindo errado
                    // std::cout << "[DEBUG] Parametro " << name << " nao e numero: " << val.dump() << std::endl;
                    return 0.0;
                }
                return val.get<double>();
            }
            
            // Caso 2: Coluna de Indicador (atr)
            if (type == "col") {
                std::string name = expr.value("name", "");
                if (data.find(name) == data.end()) {
                    return 0.0; // Retorna 0 em vez de crashar se a coluna não existir
                }
                int shift = expr.value("shift", 0);
                size_t idx = i - shift;
                
                // Proteção contra índice negativo ou fora do range
                if (i >= (size_t)shift && idx < data.at(name).size()) {
                    double val = data.at(name)[idx];
                    return std::isnan(val) ? 0.0 : val; // Trata NaN vindo do Python
                }
                return 0.0;
            }
            
            // Caso 3: Operação Matemática (Aqui resolve o ATR * SL_PERC)
            if (type == "operation") {
                // Chamada recursiva para resolver os dois lados da conta
                double left = resolve_expression(expr["left"], data, params, i);
                double right = resolve_expression(expr["right"], data, params, i);
                std::string op = expr.value("op", "");

                if (op == "*") return left * right;
                if (op == "+") return left + right;
                if (op == "-") return left - right;
                if (op == "/") return (right != 0) ? left / right : 0.0;
            }
        }
    } catch (const std::exception& e) {
        std::cout << "[DEBUG] ERRO NA RESOLUCAO: " << e.what() << " | JSON: " << expr.dump() << std::endl;
    }
    return 0.0;
}

// 2. Ajuste a evaluate_value para chamar a resolve_expression
double evaluate_value(const json& rules, const std::map<std::string, std::vector<double>>& data, const json& current_params, size_t i) {
    if (rules.is_null()) return 0.0;
    
    // Se for um array (como o Python envia [rule]), pegamos o primeiro
    if (rules.is_array()) {
        if (rules.empty()) return 0.0;
        return resolve_expression(rules[0], data, current_params, i);
    }
    
    // Se for o objeto direto
    return resolve_expression(rules, data, current_params, i);
}

// Rules selector
bool evaluate_signals(const json& rules, const std::map<std::string, std::vector<double>>& data, const json& current_params, size_t i) {
    if (!rules.is_array() || rules.empty()) return false;

    for (const auto& cond : rules) {
        std::string col_a = cond.value("a", "");
        int shift_a = cond.value("shift_a", 0);
        std::string op = cond.value("op", "");

        // Proteção de índice: i é o ponto de referência, shift_a olha para trás
        if (data.find(col_a) == data.end() || i < (size_t)shift_a) return false;
        double val_a = data.at(col_a)[i - shift_a];
        
        double val_b = 0.0;
        if (cond.contains("expr") && !cond["expr"].is_null()) {
            val_b = resolve_expression(cond["expr"], data, current_params, i);
        } else {
            std::string b_type = cond.value("b", "");
            if (b_type == "const") {
                if (cond.contains("val") && cond["val"].is_number()) val_b = cond["val"].get<double>();
            } else if (b_type == "param") {
                val_b = current_params.value(cond.value("name", ""), 0.0);
            } else {
                int shift_b = cond.value("shift_b", 0);
                if (data.find(b_type) != data.end() && i >= (size_t)shift_b) {
                    val_b = data.at(b_type)[i - shift_b];
                }
            }
        }

        if (op == ">" && !(val_a > val_b)) return false;
        if (op == "<" && !(val_a < val_b)) return false;
        if (op == ">=" && !(val_a >= val_b)) return false;
        if (op == "<=" && !(val_a <= val_b)) return false;
        if (op == "==" && !(val_a == val_b)) return false;
    }
    return true;
}



json Backtest::run_simulation(const std::string& header, 
                              const std::map<std::string, std::vector<double>>& data,
                              const std::vector<std::string>& datetime,
                              const nlohmann::json& sim,
                              const nlohmann::json& exec_settings) {
    
    // LOG inicial de verificação de indicadores
    // --- DEBUG: Print das últimas 30 linhas de todos os indicadores/preços ---
    std::cout << "\n[C++ DEBUG] Exibindo últimas 30 linhas de todos os dados (" << header << "):" << std::endl;

    if (!data.empty()) {
        // Pegamos o tamanho do primeiro vetor para referência (ex: 'close')
        size_t total_rows = data.begin()->second.size();
        size_t start_row = (total_rows > 30) ? (total_rows - 30) : 0;

        // Cabeçalho das colunas
        std::cout << "Row\t";
        for (auto const& [col_name, _] : data) {
            std::cout << col_name << "\t\t";
        }
        std::cout << "\n" << std::string(80, '-') << std::endl;

        // Loop pelas linhas
        for (size_t i = start_row; i < total_rows; ++i) {
            std::cout << i << "\t";
            for (auto const& [col_name, vec] : data) {
                // Formatação para 5 casas decimais
                if (i < vec.size()) {
                    printf("%.5f\t", vec[i]);
                } else {
                    std::cout << "N/A\t\t";
                }
            }
            std::cout << "\n";
        }
    } else {
        std::cout << "Warning: Data map is empty!" << std::endl;
    }
    std::cout << std::string(80, '=') << "\n" << std::endl;

    std::vector<Trade> trades;
    std::vector<Trade> active_trades;
    
    // PnL Temporário para validação cruzada (Aritmético simples)
    double temp_cumulative_pnl = 0.0;
    int counter = 20;
    int counter_max = counter + 30;

    try {
        const auto& open = data.at("open");
        const auto& high = data.at("high");
        const auto& low = data.at("low");
        const auto& close = data.at("close");
        size_t n_bars = datetime.size();

        std::vector<int> time_ints(n_bars);
        for (size_t i = 0; i < n_bars; ++i) {
            time_ints[i] = std::stoi(datetime[i].substr(11, 2)) * 10000 + 
                std::stoi(datetime[i].substr(14, 2)) * 100 + 
                std::stoi(datetime[i].substr(17, 2));
        }

        json params = sim.value("params", json::object());
        json rules = sim.value("rules", json::object());

        json rules_entry_long = rules.value("entry_long", json::array());
        json rules_entry_short = rules.value("entry_short", json::array());
        int nb_long = params.value("exit_nb_long", 0);
        int nb_short = params.value("exit_nb_short", 0);
        int exit_nb_only_if_pnl_is = params.value("exit_nb_only_if_pnl_is", 0);
        json rules_tf_long = rules.value("exit_tf_long", json::array());
        json rules_tf_short = rules.value("exit_tf_short", json::array());
        json rules_sl_long = rules.value("exit_sl_long_price", json::array());
        json rules_sl_short = rules.value("exit_sl_short_price", json::array());
        json rules_tp_long = rules.value("exit_tp_long_price", json::array());
        json rules_tp_short = rules.value("exit_tp_short_price", json::array());
        json be_pos_long_signal = rules.value("be_pos_long_signal", json::array());
        json be_pos_short_signal = rules.value("be_pos_short_signal", json::array());
        json be_neg_long_signal = rules.value("be_neg_long_signal", json::array());
        json be_neg_short_signal = rules.value("be_neg_short_signal", json::array());
        json be_pos_long_value = rules.value("be_pos_long_value", json::array());
        json be_pos_short_value = rules.value("be_pos_short_value", json::array());
        json be_neg_long_value = rules.value("be_neg_long_value", json::array());
        json be_neg_short_value = rules.value("be_neg_short_value", json::array());
        bool has_be_pos_l_sig = !be_pos_long_signal.empty();
        bool has_be_pos_s_sig = !be_pos_short_signal.empty();
        bool has_be_neg_l_sig = !be_neg_long_signal.empty();
        bool has_be_neg_s_sig = !be_neg_short_signal.empty();
        bool has_be_pos_l_val = !be_pos_long_value.empty();
        bool has_be_pos_s_val = !be_pos_short_value.empty();
        bool has_be_neg_l_val = !be_neg_long_value.empty();
        bool has_be_neg_s_val = !be_neg_short_value.empty();

        auto strat_num_pos = exec_settings.value("strat_num_pos", json::array({1,1}));
        int max_long = strat_num_pos[0].get<int>();
        int max_short = strat_num_pos[1].get<int>();

        bool hedge_enabled = exec_settings.value("hedge", false);
        std::string asset_base_name = header + "_" + sim["id"].get<std::string>();
        std::string order_type = exec_settings.value("order_type", "market");
        double offset = exec_settings.value("offset", 0.0);

        bool is_daytrade = exec_settings.value("day_trade", false);
        std::vector<int> close_days = exec_settings.value("day_of_week_close_and_stop_trade", std::vector<int>{});
        std::vector<int> bar_days(n_bars);
        for (size_t i=0; i<n_bars; ++i) {
            std::tm tm = {};
            std::istringstream ss(datetime[i]);
            ss >> std::get_time(&tm, "%Y-%m-%d");
            std::mktime(&tm);
            bar_days[i] = tm.tm_wday;
        }

        //export_to_csv("debug_market_data.csv", datetime, data);



        for (size_t i = 1; i < n_bars; ++i) {
            // --- 1. EXIT LOGIC (Baseado no Open do candle atual para evitar look-ahead) ---
            auto it = active_trades.begin();
            while (it != active_trades.end()) {
                bool is_long = (it->lot_size.value_or(0.0) > 0.0);
                bool closed = false;
                std::string reason = "";
                
                // TF - Trend Following
                if (evaluate_signals(is_long ? rules_tf_long : rules_tf_short, data, params, i-1)) {
                    reason = "TF"; closed = true;
                }
                
                // NB - Number of Bars
                if (!closed) {
                    int target_nb = is_long ? nb_long : nb_short;
                    
                    if (target_nb > 0 && it->bars_held >= target_nb) {
                        bool can_exit_by_pnl = false;

                        if (exit_nb_only_if_pnl_is == 0) { can_exit_by_pnl = true; }
                        else {
                            double current_price = open[i];
                            double entry_p = *it->entry_price;
                            double pnl = (current_price - entry_p) / entry_p * (is_long ? 1.0 : -1.0);

                            if (exit_nb_only_if_pnl_is > 0 && pnl > 0) can_exit_by_pnl = true;
                            else if (exit_nb_only_if_pnl_is < 0 && pnl < 0) can_exit_by_pnl = true;
                        }

                        if (can_exit_by_pnl) { reason = "NB"; closed = true; }
                    }
                }

                // DT - Day Trade
                if (!closed && is_daytrade && i+1 < n_bars) {
                    if (time_ints[i+1] < time_ints[i]) {
                        reason = "DT"; closed = true;
                    }
                }

                // WC - Weekday Close
                if (!closed && !close_days.empty()) {
                    int today = bar_days[i];
                    if (std::find(close_days.begin(), close_days.end(), today) != close_days.end()) {
                        reason = "WC"; closed = true;
                    }
                }

                // EF - End of File
                if (!closed && i == n_bars - 1) {
                    reason = "EF"; closed = true;
                }

                // BE - SL/TP to Break Even
                if (!closed) {
                    double entry_p = *it->entry_price;
                    double current_p = open[i];

                    // BE Positive
                    if (it->stop_loss != entry_p && (is_long ? (has_be_pos_l_sig || has_be_pos_l_val) : (has_be_pos_s_sig || has_be_pos_s_val))) {
                        bool trig_be_pos = false;

                        if (is_long ? has_be_pos_l_sig : has_be_pos_s_sig) {
                            trig_be_pos = evaluate_signals(is_long ? be_pos_long_signal : be_pos_short_signal, data, params, i-1);
                        }

                        if (!trig_be_pos && (is_long ? has_be_pos_l_val : has_be_pos_s_val)) {
                            double v = evaluate_value(is_long ? be_pos_long_value : be_pos_short_value, data, params, i-1);
                            if (v > 0) {
                                trig_be_pos = is_long ? (current_p >= entry_p + v) : (current_p <= entry_p - v);
                            }
                        }

                        if (trig_be_pos) {
                            if (is_long ? (current_p > entry_p) : (current_p < entry_p)) {
                                it->stop_loss = entry_p;
                            }
                        }
                    }

                    // BE Negative
                    if (it->take_profit != entry_p && (is_long ? (has_be_neg_l_sig || has_be_neg_l_val) : (has_be_neg_s_sig || has_be_neg_s_val))) {
                        bool trig_be_neg = false;

                        if (is_long ? has_be_neg_l_sig : has_be_neg_s_sig) {
                            trig_be_neg = evaluate_signals(is_long ? be_neg_long_signal : be_neg_short_signal, data, params, i-1);
                        }

                        if (!trig_be_neg && (is_long ? has_be_neg_l_val : has_be_neg_s_val)) {
                            double v = evaluate_value(is_long ? be_neg_long_value : be_neg_short_value, data, params, i-1);
                            if (v > 0) {
                                trig_be_neg = is_long ? (current_p <= entry_p - v) : (current_p >= entry_p + v);
                            }
                        }
                        
                        if (trig_be_neg) {
                            if (is_long ? (current_p < entry_p) : (current_p > entry_p)) {
                                it->take_profit = entry_p;
                            }
                        }
                    }
                }



                if (closed) {
                    it->exit_price = open[i];
                    it->exit_datetime = datetime[i];
                    it->exit_reason = reason;
                    it->status = "closed";
                    
                    double entry = *it->entry_price;
                    double exit = *it->exit_price;
                    it->profit = ((exit - entry) / entry) * 100 * (is_long ? 1 : -1);
                    temp_cumulative_pnl += it->profit.value_or(0.0);

                    if (counter < counter_max) {
                        std::cout << "[EXIT ] " << (is_long ? "L" : "S") << " | " << reason << " " << datetime[i] 
                                << " | In: " << entry << " | Out: " << exit
                                << " | PnL: " << *it->profit << "% | Acc: " << temp_cumulative_pnl << "%" << std::endl;
                        counter+=1;
                    }

                    trades.push_back(std::move(*it));
                    it = active_trades.erase(it);
                } else { ++it; }
            }
            
            // --- 2. ENTRY LOGIC (Sinal no candle anterior i-1, entrada no Open de i) ---
            bool day_is_blocked = false;
            if (!close_days.empty()) {
                int today = bar_days[i];
                day_is_blocked = std::find(close_days.begin(), close_days.end(), today) != close_days.end();
            }

            if (!day_is_blocked) {
                bool signal_long = evaluate_signals(rules_entry_long, data, params, i - 1);
                bool signal_short = evaluate_signals(rules_entry_short, data, params, i - 1);

                if (signal_long || signal_short) {
                    int current_longs = 0;
                    int current_shorts = 0;

                    for (const auto& t : active_trades) {
                        double lot = t.lot_size.value_or(0.0);
                        if (lot > 0) current_longs++;
                        else if (lot < -0) current_shorts++;
                    }

                    auto execute_entry_internal = [&](bool is_long) {
                        Trade t;
                        t.id = generate_id();
                        t.asset = asset_base_name;
                        t.entry_datetime = datetime[i];
                        t.status = "open";
                        double entry_price = open[i];
                        t.entry_price = entry_price;
                        t.max_fav_price = entry_price;
                        t.max_adv_price = entry_price;
                        t.bars_held = 0;
                        t.lot_size = calculate_lot_size(entry_price, is_long);

                        const json& sl_rules = is_long ? rules_sl_long : rules_sl_short;
                        const json& tp_rules = is_long ? rules_tp_long : rules_tp_short;
                        double sl_dist = evaluate_value(sl_rules, data, params, i);
                        double tp_dist = evaluate_value(tp_rules, data, params, i);
                        
                        double sl = is_long ? (entry_price - sl_dist) : (entry_price + sl_dist);
                        double tp = is_long ? (entry_price + tp_dist) : (entry_price - tp_dist);

                        if (sl_dist > 0) t.stop_loss = sl;
                        if (tp_dist > 0) t.take_profit = tp;

                        if (counter < counter_max) {
                            std::cout << "[ENTRY] " << (is_long ? "L" : "S") << " |    " << datetime[i] 
                                    << " | Price: " << entry_price << " | TP: " << tp << " | SL: " << sl << std::endl;
                            counter++;
                        }
                        active_trades.push_back(std::move(t));
                    };

                    if (signal_long) {
                        bool hedge_ok = hedge_enabled || (current_shorts == 0);
                        if (current_longs < max_long && hedge_ok) execute_entry_internal(true);
                    }

                    if (signal_short) {
                        bool hedge_ok = hedge_enabled || (current_longs == 0);
                        if (current_shorts < max_short && hedge_ok) execute_entry_internal(false);
                    }
                }
            }

            // --- 3. INTRA-CANDLE EXIT (SL/TP) ---
            it = active_trades.begin();
            while (it != active_trades.end()){
                double sl = it->stop_loss.value_or(0.0);
                double tp = it->take_profit.value_or(0.0);
                bool is_long = (it->lot_size.value_or(0.0) > 0.0);
                bool closed = false;
                std::string reason = "";

                // Checagem de Stop Loss / Take Profit usando High/Low do candle i
                if (is_long) {
                    if (sl > 0.0 && low[i] <= sl) {
                        it->exit_price = sl; reason = "SL"; closed = true; 
                    }
                    else if (tp > 0.0 && high[i] >= tp){
                        it->exit_price = tp; reason = "TP"; closed = true;
                    }
                } else {
                    if (sl > 0.0 && high[i] >= sl) {
                        it->exit_price = sl; reason = "SL"; closed = true;
                    }
                    else if (tp > 0.0 && low[i] <= tp){
                        it->exit_price = tp; reason = "TP"; closed = true;
                    }
                }

                if (closed) {
                    double entry = *it->entry_price;
                    it->exit_datetime = datetime[i];
                    it->exit_reason = reason;
                    it->status = "closed";

                    it->profit = ((*it->exit_price - entry) / entry) * 100.0 * (is_long ? 1.0 : -1.0);
                    temp_cumulative_pnl += *it->profit;

                    if (counter < counter_max) { 
                        std::cout << "[EXIT ] " << (is_long ? "L" : "S") << " | " << reason << " " << datetime[i] << " | PnL: " << *it->profit 
                                << "% | Acc: " << temp_cumulative_pnl << "%" << std::endl;
                        counter+=1;
                    }

                    trades.push_back(std::move(*it));
                    it = active_trades.erase(it);
                }
                else {
                    // Atualiza MFE/MAE se o trade continuar aberto
                    double current_high = high[i];
                    double current_low = low[i];

                    if (is_long) {
                        if (current_high > *it->max_fav_price) it->max_fav_price = current_high;
                        if (current_low < *it->max_adv_price) it->max_adv_price = current_low;
                    } else {
                        if (current_low < *it->max_fav_price) it->max_fav_price = current_low;
                        if (current_high > *it->max_adv_price) it->max_adv_price = current_high;
                    }

                    it->bars_held = (it->bars_held.value_or(0)) + 1;
                    ++it;
                }
            }
        }

    } catch (const std::exception& e) { std::cerr << "[C++ Backtest Error]: " << e.what() << std::endl; }

    std::cout << "[FINISH] Total Trades: " << trades.size() << " | PnL Validado: " << temp_cumulative_pnl << "%" << std::endl;
    return trades_to_json(trades);
}


// Placeholder
double calculate_lot_size(double price_entry, bool is_long_trade) {
    return (is_long_trade) ? 1.0 : -1.0;
}



