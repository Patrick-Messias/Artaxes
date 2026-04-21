from dataclasses import dataclass
from MoneyManager import MoneyManager, MoneyManagerParams

@dataclass
class ModelMoneyManagerParams(MoneyManagerParams):
    pass

class ModelMoneyManager(MoneyManager): # Manages Model's risk and money management
    def __init__(self, mmm_params: ModelMoneyManagerParams): # PMM(Portfolio) > MMM(Model) > SMM(Strat)
        super().__init__(mmm_params)

#||=========================================================================================||

    def _default_pre_compute(self, global_assets, timeline, aggr_ret, indicator_pool, param_sets) -> dict:
        # By Default doesn't calculate anything else, but can be used to prepare signals or other stuff != indicators
        return indicator_pool
          
    # ── Every Datetime [i] ───────────────────────────────────────────────

    def _default_allocate(self, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key):
        return hierarchy, indicator_pool, sim_data, port_returns

    def _default_size(self, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key):
        return hierarchy, indicator_pool, sim_data, port_returns

    def _default_risk_guard(self, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key):
        return hierarchy, indicator_pool, sim_data, port_returns

    def _default_main(self, step_dt, hierarchy: dict, indicator_pool: dict, port_returns: dict, key) -> bool:

        sim_data = self.get_data(key=key, lookback=self.reb_lookback, data_type="aggr", side="both")

        hierarchy, indicator_pool, sim_data, port_returns = self.allocate(step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)
        hierarchy, indicator_pool, sim_data, port_returns = self.size(step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)
        hierarchy, indicator_pool, sim_data, port_returns = self.risk_guard(step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)

        return hierarchy

#||=========================================================================================||