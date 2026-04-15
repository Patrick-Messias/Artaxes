from dataclasses import dataclass, field
from SystemManager import SystemManager, SystemManagerParams
from typing import Optional, Callable, Dict, List

@dataclass
class ModelSystemManagerParams(SystemManagerParams):
    model_hierarchy: dict = field(default_factory=lambda: {"order_by": 'highest', "metric": 'profit_perc'})
    rebalance_frequency: str = 'weekly'
    close_open_trades_on_rebalance: bool = False

class ModelSystemManager(SystemManager): # Manages portfolio's model hierarchy 
    def __init__(self, msm_params: ModelSystemManagerParams):
        super().__init__(msm_params) # SystemManager attributes init
        
        self.model_hierarchy = dict(msm_params.model_hierarchy)
        self.rebalance_frequency = msm_params.rebalance_frequency
        self.close_open_trades_on_rebalance = msm_params.close_open_trades_on_rebalance

#||=========================================================================================||

    def _default_pre_compute(self, global_assets, timeline, sim_data, aggr_ret, indicator_pool, param_sets) -> dict:
        # By Default doesn't calculate anything else, but can be used to prepare signals or other stuff != indicators
        return indicator_pool, sim_data
                       
    def _default_rank(self, step_dt, hierarchy: dict, indicator_pool: dict, op_data: dict, port_returns: dict) -> Dict[str, float]:
        return hierarchy, indicator_pool, op_data, port_returns

    def _default_filter(self, step_dt, hierarchy: dict, indicator_pool: dict, op_data: dict, port_returns: dict) -> List[str]:
        return hierarchy, indicator_pool, op_data, port_returns # By default doesn't filter out any model

    def _default_rebalance(self, step_dt, hierarchy: dict, indicator_pool: dict, op_data: dict, port_returns: dict) -> List[str]:
        return hierarchy, indicator_pool, op_data, port_returns

    # ── Every Datetime [i] ───────────────────────────────────────────────

    def main(self, step_dt, hierarchy: dict, indicator_pool: dict, op_data: dict, port_returns: dict) -> bool:
        # Called every datetime for each model and asset
        # Returns True if model can operate now
        return self._call(self._fn_main, self._default_main, step_dt, hierarchy, indicator_pool, op_data, port_returns)
    
    def _default_main(self, step_dt, hierarchy: dict, indicator_pool: dict, op_data: dict, port_returns: dict) -> bool:

        hierarchy, indicator_pool, op_data, port_returns = self.rank(step_dt, hierarchy, indicator_pool, op_data, port_returns)
        hierarchy, indicator_pool, op_data, port_returns = self.filter(step_dt, hierarchy, indicator_pool, op_data, port_returns)
        hierarchy, indicator_pool, op_data, port_returns = self.rebalance(step_dt, hierarchy, indicator_pool, op_data, port_returns)

        return hierarchy

#||=========================================================================================||








    """ # NOTE Não deletar abaixo, exemplo de MSM
        FinancialWisdom_Explosive_Stock_Asset_Rank_System

        - price < 100.0
        - market_cap < 10 Billion
        - Tech and Biotech Sectors (Sector Allocation Distribution)
        - Momentum Precedes Explosive Moves (Forecasts and Technical Indicators)
        - Weekly Timeframe

    """


