from dataclasses import dataclass
from MoneyManager import MoneyManager, MoneyManagerParams

@dataclass
class PortfolioMoneyManagerParams(MoneyManagerParams):
    pass

class PortfolioMoneyManager(MoneyManager): # Manages Model's risk and money management
    def __init__(self, pmm_params: PortfolioMoneyManagerParams): # PMM(Portfolio) > MMM(Model) > MMA(Strat)
        super().__init__(pmm_params)

    def calculate_portfolio_capital_allocation(self): # Rebalances capital allocation between Models of the Portfolio
        pass