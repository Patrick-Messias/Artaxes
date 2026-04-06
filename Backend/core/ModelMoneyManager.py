from dataclasses import dataclass
from MoneyManager import MoneyManager, MoneyManagerParams

@dataclass
class ModelMoneyManagerParams(MoneyManagerParams):
    pass

class ModelMoneyManager(MoneyManager): # Manages Model's risk and money management
    def __init__(self, mmm_params: ModelMoneyManagerParams): # PMM(Portfolio) > MMM(Model) > SMM(Strat)
        super().__init__(mmm_params)

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