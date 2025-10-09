from dataclasses import dataclass
from MoneyManager import MoneyManager, MoneyManagerParams

@dataclass
class StratMoneyManagerParams(MoneyManagerParams):

class StratMoneyManager(MoneyManager): # Manages Strat's risk and money management
    def __init__(self, smm_params: StratMoneyManagerParams): # PMM(Portfolio) > MMM(Model) > SMM(Strat)
        super().__init__(smm_params)

    def calculate_strat_capital_allocation(self, capital: float, strat: Strat): 
        return strat