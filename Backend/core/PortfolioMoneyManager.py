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

    def _default_pre_compute(self, global_assets, timeline, aggr_ret, indicator_pool, param_sets) -> dict:
        # By Default doesn't calculate anything else, but can be used to prepare signals or other stuff != indicators
        return indicator_pool
          
    # ── Every Datetime [i] ───────────────────────────────────────────────

    def _default_allocate(self, i, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key):
        return hierarchy, indicator_pool, sim_data, port_returns

    def _default_size(self, i, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key):
        return hierarchy, indicator_pool, sim_data, port_returns

    def _default_risk_guard(self, i, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key):
        return hierarchy, indicator_pool, sim_data, port_returns

    def _default_main(self, i, step_dt, hierarchy: dict, indicator_pool: dict, port_returns: dict, key) -> bool:

        # Default uses aggr of models for Portfolio Level
        sim_data = self.get_data(key=key, lookback=self.reb_lookback, data_type="aggr", side="both")

        hierarchy, indicator_pool, sim_data, port_returns = self.allocate(i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)
        hierarchy, indicator_pool, sim_data, port_returns = self.size(i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)
        hierarchy, indicator_pool, sim_data, port_returns = self.risk_guard(i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)

        return hierarchy

#||=========================================================================================||










