from dataclasses import dataclass, field
from SystemManager import SystemManager, SystemManagerParams
from typing import Optional, Callable, Dict, List

@dataclass
class StratSystemManagerParams(SystemManagerParams):
    strat_hierarchy: dict = field(default_factory=lambda: {"order_by": 'highest', "metric": 'profit_perc'})
    rebalance_frequency: str = 'weekly'
    strat_lookback_n: int = 1
    close_open_trades_on_rebalance: bool = False

class StratSystemManager(SystemManager): 
    def __init__(self, ssm_params: SystemManagerParams):
        super().__init__(ssm_params) 
        
        self.strat_hierarchy = dict(ssm_params.strat_hierarchy)
        self.rebalance_frequency = ssm_params.rebalance_frequency
        self.strat_lookback_n = ssm_params.strat_lookback_n
        self.close_open_trades_on_rebalance = ssm_params.close_open_trades_on_rebalance

#||=========================================================================================||

    def _default_pre_compute(self, global_assets, timeline, aggr_ret, indicator_pool, param_sets, manager_level_key) -> dict:
        # By Default doesn't calculate anything else, but can be used to prepare signals or other stuff != indicators
        return indicator_pool
                       
    def _default_rank(self, i, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key) -> Dict[str, float]:
        return hierarchy, indicator_pool, sim_data, port_returns

    def _default_filter(self, i, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key) -> List[str]:
        return hierarchy, indicator_pool, sim_data, port_returns # By default doesn't filter out any model

    def _default_rebalance(self, i, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key) -> List[str]:
        return hierarchy, indicator_pool, sim_data, port_returns

    # ── Every Datetime [i] ───────────────────────────────────────────────

    def _default_main(self, i, step_dt, hierarchy: dict, indicator_pool: dict, port_returns: dict, key) -> bool:

        sim_data = self.get_data(key=key, lookback=self.reb_lookback, data_type="aggr", side="both")

        hierarchy, indicator_pool, sim_data, port_returns  = self.rank(i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)
        hierarchy, indicator_pool, sim_data, port_returns  = self.filter(i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)
        hierarchy, indicator_pool, sim_data, port_returns  = self.rebalance(i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)

        return hierarchy

#||=========================================================================================||


















