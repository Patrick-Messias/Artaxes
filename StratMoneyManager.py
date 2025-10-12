from dataclasses import dataclass, field
from Trade import Trade
from MoneyManager import MoneyManager, MoneyManagerParams

@dataclass
class StratMoneyManagerParams(MoneyManagerParams):
    position_allocation: dict = field(default_factory=lambda: {"type": "percentage", "from": "capital", "method": "regular", "compounding": False, "leverage": 1.0}) # 'percentage', 'kelly', 'confidence' | 'balance', 'equity' | 'regular', 'dynamic'
    position_sizing: dict = field(default_factory=lambda: {"position_risk": 0.01, "position_risk_min": 0.001, "position_risk_max": 0.05})

    # Usar variávle abaixo nas def como valor default 
    trade_max_num_open: int = 1
    trade_min_num_analysis: int = 100

class StratMoneyManager(MoneyManager): # Manages Strat's risk and money management
    def __init__(self, smm_params: StratMoneyManagerParams): # PMM(Portfolio) > MMM(Model) > SMM(Strat)
        super().__init__(smm_params)

        # Trade Risk - Trade Management
        self.position_allocation = smm_params.position_allocation
        self.position_sizing = smm_params.position_sizing
        self.trade_max_num_open = smm_params.trade_max_num_open
        self.trade_min_num_analysis = smm_params.trade_min_num_analysis

    def calculate_strat_capital_allocation(self, capital: float, strat: 'Strat'): 
        from Strat import Strat, StratParams
        
        return strat

    def calculate_kelly_criterion(self, trades: dict[Trade], weight: int=0.1):
        return None