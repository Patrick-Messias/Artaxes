from dataclasses import dataclass, field
#from Backend.core import Asset
from MoneyManager import MoneyManager, MoneyManagerParams
from typing import Optional, Dict, Literal, Callable, List
import polars as pl

@dataclass
class PortfolioMoneyManagerParams(MoneyManagerParams):
    # Allocation
    alo_allocation: Optional[Dict[str, float]]=None # Ex: {"Model_A": 0.5, "Model_B": 0.3, "Model_C": 0.2} -> 50% do capital para Model_A, 30% para Model_B e 20% para Model_C

    # Rebalancing
    reb_metric: Literal["pnl", "pnl_dd", "sharpe"] = "pnl" # Metric used for performance-based rebalancing (if reb_method == "performance")
    reb_method: Literal["fixed", "equal_weight", "risk_parity", "performance"] = "fixed"
    reb_deviation_func: Optional[Dict[str, Callable]] = None # Function that defines the deviation threshold needed for rebalancing (e.g., 5% deviation from target allocation)

class PortfolioMoneyManager(MoneyManager): # Manages Model's risk and money management
    def __init__(self, pmm_params: PortfolioMoneyManagerParams): # PMM(Portfolio) > MMM(Model) > MMA(Strat)
        super().__init__(pmm_params)
        self.alo_allocation = pmm_params.alo_allocation

        self.reb_metric = pmm_params.reb_metric
        self.reb_method = pmm_params.reb_method
        self.reb_deviation_func = pmm_params.reb_deviation_func

        self._pre_cache: Dict = {}

#||=========================================================================================||

    def _default_pre_compute(self, global_assets, timeline, sim_data, aggr_ret, indicator_pool, param_sets) -> dict:

        # By Default doesn't calculate anything else, but can be used to prepare signals or other stuff != indicators
        
        return indicator_pool, sim_data
          
    def _default_allocate(self):
        pass

    def _default_size(self):
        pass

    def _default_risk_guard(self):
        pass

    # ── Every Datetime [i] ───────────────────────────────────────────────

    def main(self, step_dt, hierarchy: dict, op_data: dict, port_returns: dict) -> bool:
        # Called every datetime for each model and asset
        # Returns True if model can operate now
        return self._call(self._fn_main, self._default_main, step_dt, hierarchy, op_data, port_returns)
    
    def _default_main(self, step_dt, hierarchy: dict, op_data: dict, port_returns: dict) -> bool:

        # Calculates Live Indicators

        # Rebalances

        return hierarchy

#||=========================================================================================||




# Indicators

# MCMC
# HMC
# VaR







