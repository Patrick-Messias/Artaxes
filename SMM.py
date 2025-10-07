from typing import Dict, Optional, Union, List, Callable
from dataclasses import dataclass, field
from Money_Management_Algorithm import Money_Management_Algorithm
import Indicator

@dataclass
class Strategy_Management_Algorithm_Params:
    """Parâmetros para configurar o Money Management"""
    name: str='unnamed_mma'

class Strategy_Management_Algorithm(Money_Management_Algorithm): # Manages Strat's risk and money management
    def __init__(self, mma_params: Strategy_Management_Algorithm_Params): # PMM(Portfolio) > MMM(Model) > MMA(Strat)
        super().__init__()
        self.name: mma_params.name


    def calculate_trade_risk_perc(self, trades: list[Trade], type_management: str, risk_default: float=0.005, risk_min: float=0.001, risk_max: float=0.05,
                            min_quant_trades=100, confidence_level=0.5, kelly_weight=0.1):
        
        if len(trades) < min_quant_trades:
            return risk_default
            
        if type_management == 'regular':
            return risk_default
            
        elif type_management == 'kelly':
            wins = [trade for trade in trades if trade.profit > 0]
            losses = [trade for trade in trades if trade.profit <= 0]
            
            if not wins or not losses:
                return risk_default
                
            win_prob = len(wins) / len(trades)
            avg_win = sum(trade.profit_r for trade in wins) / len(wins)
            avg_loss = abs(sum(trade.profit_r for trade in losses) / len(losses))
            
            kelly = win_prob - ((1 - win_prob) / (avg_win / avg_loss))
            kelly *= kelly_weight
            
            return max(min(kelly, risk_max), risk_min)
            
        elif type_management == 'confidence':
            sorted_returns = sorted(trade.profit_r for trade in trades)
            index = int(len(sorted_returns) * confidence_level)
            conf_return = sorted_returns[index]
            
            if conf_return <= 0:
                return risk_min
            return max(min(conf_return * risk_default, risk_max), risk_min)
            
        else:
            return risk_default

    def calculate_trade_lot_size(self, asset_metrics: dict, stop_diff: float, balance: float, risk_percent: float):
        fin_risk = balance * risk_percent
        price_risk = stop_diff * asset_metrics['tick_fin_val']
        
        if price_risk == 0:
            return 0
            
        lot_size = fin_risk / price_risk
        lot_size = max(round(lot_size / asset_metrics['min_lot']) * asset_metrics['min_lot'], asset_metrics['min_lot'])
        
        max_leverage = balance * asset_metrics['leverage']
        max_lot = max_leverage / asset_metrics['lot_value']
        
        return min(lot_size, max_lot)

    def calculate_trade_drawdown_limits(self, trades: list[Trade], MMA: Money_Management_Algorithm, asset_data: pd.DataFrame=None): # Adjusts drawdown limits with Strat's methods

        # Checks drawdown method and any special mma_rules

        # Calculates new drawdown limits with trade list and or data for only used drawdown limits (drawdown_global)

        # If new data calculated change MMA values for new ones

        return MMA



