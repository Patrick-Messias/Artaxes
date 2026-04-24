from dataclasses import dataclass, field
from SystemManager import SystemManager, SystemManagerParams
from typing import List, Optional, Dict, Literal, Callable, Union
#from Backend.core import Asset
import polars as pl, numpy as np

@dataclass
class PortfolioSystemManagerParams(SystemManagerParams):
    model_hierarchy: Dict = field(default_factory=lambda: {
        "order_by": "highest",
        "metric":   "pnl"
    })
    max_active_models: Optional[int] = None # Max number of active models in the portfolio at any given time (if None, no limit)

    # Rebalancing
    reb_metric: Literal["pnl", "pnl_dd", "sharpe"] = "pnl" # Metric used for performance-based rebalancing (if reb_method == "performance")
    reb_method: Literal["fixed", "equal_weight", "risk_parity", "performance"] = "fixed"
    reb_deviation_func: Optional[Dict[str, Callable]] = None # Only rebalance if (ex: Portfolio std deviated "x" std from mean)
    reb_closes_open_trades_on_rebalance: bool = False # NOTE add this only to StratSystemManager

class PortfolioSystemManager(SystemManager): # Manages portfolio's model hierarchy 
    def __init__(self, psm_params: PortfolioSystemManagerParams):
        super().__init__(psm_params)

        self.reb_metric                         = psm_params.reb_metric
        self.model_hierarchy                    = dict(psm_params.model_hierarchy)
        self.max_active_models                  = psm_params.max_active_models
        self.reb_method                         = psm_params.reb_method
        self.reb_closes_open_trades_on_rebalance = psm_params.reb_closes_open_trades_on_rebalance

#||=========================================================================================||

    def _default_pre_compute(self, global_assets, timeline, aggr_ret, indicator_pool, param_sets, manager_level_key) -> dict:
        # By Default doesn't calculate anything else, but can be used to prepare signals or other stuff != indicators
        return indicator_pool
                       
    # ── Every Datetime [i] ───────────────────────────────────────────────
    
    def _default_rank(self, i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key) -> dict:
        # Ranks models by metric defined in model_hierarchy. Returns dict[model_name: score]
        # for m_key in hierarchy['models'].keys():
        #     m_pnl = sim_data.get(m_key, {}).get('both', {}).get('data')
        #     if m_pnl is not None:
        #         hierarchy['models'][m_key]['score'] = float(m_pnl[i])
        
        return hierarchy

    def _default_filter(self, i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key) -> dict:
        # Disables models that don't pass the filter function. Returns dict with 'active' field updated
        # for m_key in hierarchy['models'].keys():
        #     score = hierarchy['models'][m_key].get('score', 0)
            
        #     # Exemplo: Desativa se o score for negativo
        #     if score < 0:
        #         hierarchy['models'][m_key]['active'] = False
            
        return hierarchy

    def _default_rebalance(self, i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key) -> dict:


        return hierarchy

    def _default_main(self, i, step_dt, hierarchy: dict, indicator_pool: dict, port_returns: dict, key) -> bool:
        
        # Default uses aggr of models for Portfolio Level
        sim_data = self.get_data(key=key, lookback=self.reb_lookback, data_type="aggr", side="both") 

        hierarchy = self.rank(i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)
        hierarchy = self.filter(i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)
        hierarchy = self.rebalance(i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)

        return hierarchy
   
#||=========================================================================================||

    """ Dt execution framework

    1. Check current tradable Models
    -> REBALANCE

    2. New Rank generated with updated data
    3. Needs to remove any Models? if yes then close or keep positions open by SM rules? != MM rules
    4. Needs to add any Models? 
    

    """


















