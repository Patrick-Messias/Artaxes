#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

struct Candle {
    std::string datetime;
    double open;
    double high;
    double low;
    double close;
    double volume;
    double spread;
};

struct Trade {
    std::string entryTime;
    std::string exitTime;
    double entryPrice;
    double exitPrice;
    std::string type;
    std::string exitReason;
    double profitPcnt;
};

class Backtester {
private:
    double stopLossPcnt = 0.06;
    double takeProfitPcnt = 0.03;
    bool daytrade = true;
    std::vector<Trade> history;
    bool inPosition = false;
    Trade currentTrade;

public:
    void run(const std::vector<Candle>& data) {
        if (data.size() < 4) return;

        for (size_t i = 3; i < data.size(); ++i) {
            if (!inPosition) {
                if (data[i-1].close > data[i-1].open &&
                    data[i-2].close > data[i-2].open &&
                    data[i-3].close > data[i-3].open) {
                    openPosition(data[i], "BUY");
                }
                else if (data[i-1].close < data[i-1].open &&
                         data[i-2].close < data[i-2].open &&
                         data[i-3].close < data[i-3].open) {
                    openPosition(data[i], "SELL");
                }
            } else {
                checkExit(data[i], data[i-1], data[i-2]);
            }
        }
    }

    void openPosition(const Candle& c, std::string type) {
        inPosition = true;
        currentTrade.entryTime = c.datetime;
        currentTrade.entryPrice = c.open;
        currentTrade.type = type;
    }

    void checkExit(const Candle& current, const Candle& prev1, const Candle& prev2) {
        double price = current.close;
        double change = (currentTrade.type == "BUY") ?
                        (price / currentTrade.entryPrice - 1.0) :
                        (currentTrade.entryPrice / price - 1.0);
        bool shouldExit = false;
        std::string reason = "";

        if (change >= takeProfitPcnt) {
            shouldExit = true;
            reason = "TP";
        }
        else if (change <= -stopLossPcnt) {
            shouldExit = true;
            reason = "SL";
        }
        else if (currentTrade.type == "BUY" && prev1.close < prev1.open && prev2.close < prev2.open) {
            shouldExit = true;
            reason = "TF";
        }
        else if (currentTrade.type == "SELL" && prev1.close > prev1.open && prev2.close > prev2.open) {
            shouldExit = true;
            reason = "TF";
        }
        else if (daytrade && current.datetime.substr(11, 5) == "23:50") {
            shouldExit = true;
            reason = "EOD";
        }

        if (shouldExit) {
            currentTrade.exitTime = current.datetime;
            currentTrade.exitPrice = price;
            currentTrade.exitReason = reason;
            currentTrade.profitPcnt = change * 100.0;
            history.push_back(currentTrade);
            inPosition = false;
        }
    }

    void printHistory() const {
        std::cout << ">>> Trading History <<<\n";
        std::cout << "Entry Date | Exit Date | type | Entry Price | Exit Price | Reason | PnL%\n";
        for (const auto& t : history) {
            std::cout << t.entryTime << " | " << t.exitTime << " | " << t.type << " | " 
                      << t.entryPrice << " | " << t.exitPrice << " | " << t.exitReason 
                      << " | " << t.profitPcnt << "%\n";
        }
    }
};

int main() {
    std::vector<Candle> mockData = {
        {"2023-01-01 10:00", 100, 105, 99, 102, 100, 0},
        {"2023-01-01 10:15", 102, 106, 101, 104, 100, 0},
        {"2023-01-01 10:30", 104, 108, 103, 107, 100, 0},
        {"2023-01-01 10:45", 107, 122, 106, 121, 100, 0},
        {"2023-01-01 11:00", 121, 120, 115, 116, 100, 0},
        {"2023-01-01 11:15", 116, 115, 110, 112, 100, 0},
        {"2023-01-01 11:30", 112, 111, 105, 108, 100, 0},
        {"2023-01-01 11:45", 108, 110, 107, 109, 100, 0},
        {"2023-01-01 12:00", 109, 112, 108, 111, 100, 0}
    };

    Backtester bt;
    bt.run(mockData);
    bt.printHistory();

    return 0;
}