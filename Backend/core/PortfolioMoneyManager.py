from dataclasses import dataclass, field
#from Backend.core import Asset
from MoneyManager import MoneyManager, MoneyManagerParams
from typing import Optional, Dict, Literal, Callable, Union
import polars as pl

@dataclass
class PortfolioMoneyManagerParams(MoneyManagerParams):
    # Allocation
    alo_allocation: Optional[Dict[str, float]]=None # Ex: {"Model_A": 0.5, "Model_B": 0.3, "Model_C": 0.2} -> 50% do capital para Model_A, 30% para Model_B e 20% para Model_C

    # Rebalancing
    reb_metric: Literal["pnl", "pnl_dd", "sharpe"] = "pnl" # Metric used for performance-based rebalancing (if reb_method == "performance")
    reb_method: Literal["fixed", "equal_weight", "risk_parity", "performance"] = "fixed"
    reb_frequency: Literal["daily", "weekly", "monthly", "quarterly", "yearly","never"] = "weekly"
    reb_lookback_n: int = 1
    reb_deviation_func: Optional[Dict[str, Callable]] = None # Function that defines the deviation threshold needed for rebalancing (e.g., 5% deviation from target allocation)

class PortfolioMoneyManager(MoneyManager): # Manages Model's risk and money management
    def __init__(self, pmm_params: PortfolioMoneyManagerParams): # PMM(Portfolio) > MMM(Model) > MMA(Strat)
        super().__init__(pmm_params)
        self.alo_allocation = pmm_params.alo_allocation

        self.reb_metric = pmm_params.reb_metric
        self.reb_method = pmm_params.reb_method
        self.reb_frequency = pmm_params.reb_frequency
        self.reb_lookback_n = pmm_params.reb_lookback_n
        self.reb_deviation_func = pmm_params.reb_deviation_func

    def calculate_model_capital_allocation(self): # Rebalances capital allocation between Models of the Portfolio
        pass

    def get_model_allocated_capital(self, model_name: str) -> float:
        # Returns absolute capital allocated to a given model
        return self.get_allocated_capital() * self.get_model_allocation(model_name)

    def get_model_allocation(self, model_name: str) -> float:
        # Returns the current capital allocation for a given model, based on the alo_allocation or alo_allocation_func
        if self.reb_method == "fixed" and self.alo_allocation:
            return self.alo_allocation.get(model_name, 0.0) 
        
        # equal weight fallback
        return 1.0 / len(self.alo_allocation) if self.alo_allocation else 0.0
    
    def rebalance(self, equity_history: Dict[str, pl.DataFrame]) -> Dict[str, float]:
        # Recalculates alocations based on historical equity of each model
        # Returns [model_name: new_fraction]
        # Called by _portfolio_simulation() at each rebalance frequency

        if self.reb_method == "equal_weight":
            n = len(equity_history)
            return {m: 1.0 / n for m in equity_history}
            
        if self.reb_method == "performance":
            # Ranks metrics based on last rebalance_lookback_n periods
            scores = {}
            for m, df in equity_history.items():
                tail = df.tail(self.reb_lookback_n)["pnl"]
                if self.reb_metric == "sharpe":
                    scores[m] = float(tail.mean() / (tail.std() + 1e-9))
                else:
                    scores[m] = float(tail.sum())

            total = sum(max(v, 0) for v in scores.values()) + 1e-9
            return {m: max(v, 0) / total for m, v in scores.items()}

        # fixed — retorna alocação definida
        return self.model_allocation or {}









# Indicators

# MCMC
# HMC
# VaR







