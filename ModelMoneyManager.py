from dataclasses import dataclass
from MoneyManager import MoneyManager, MoneyManagerParams

@dataclass
class ModelMoneyManagerParams(MoneyManagerParams):

class ModelMoneyManager(MoneyManager): # Manages Model's risk and money management
    def __init__(self, mmm_params: ModelMoneyManagerParams): # PMM(Portfolio) > MMM(Model) > SMM(Strat)
        super().__init__(mmm_params)

    def calculate_model_capital_allocation(self, capital: float, strats: dict): # Rebalances capital allocation between Strats enside the Model with PMM's capital allocated to Model
        return strats