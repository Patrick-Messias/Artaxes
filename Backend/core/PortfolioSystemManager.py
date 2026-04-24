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
        for m_key, m_info in hierarchy.items():
            if not m_info.get('active', True): 
                continue

            # Searches specific data for this model
            separate_ls = m_info.get("separate_ls", False)
            side = m_info.get("side", "both")
            if side == "both":
                side = ["long", "short"] if separate_ls else ["both"]

            print(f"m_key: {m_key}")
            print(f"sim_data keys: {list(sim_data.keys())}")

            m_data = sim_data[m_key]
            if not m_data: 
                continue

            for sd in side:
                if sd in m_data and len(m_data[sd]) > 0:
                    series = m_data[sd]
                    returns = series.flatten()

                    if len(returns) > 0:
                        score = sum(returns)
                        m_info[sd]['score'] = score
        
        return hierarchy

    def _default_filter(self, i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key) -> dict:
        # Disables models that don't pass the filter function. Returns dict with 'active' field updated
        
        for m_key, m_info in hierarchy.items():
            if not m_info.get('active', True): 
                continue

            separate_ls = m_info.get("separate_ls", False)
            side = m_info.get("side", "both")
            if side == "both":
                side = ["long", "short"] if separate_ls else ["both"]

            for sd in side:
                score = m_info[sd]['score']

                if score < -0.05:
                    m_info[sd]['weight'] = 0.0
            
        return hierarchy

    def _default_rebalance(self, i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key) -> dict:
        # Rebalances Models using HRP by default
        import numpy as np, polars as pl
        from indicators.HRP import HRP

        lookback = self.reb_lookback
        start_idx = max(0, i-lookback)

        matrix_data = {}
        model_map = {}

        # Collects and prepares matrix
        for m_key, m_info in hierarchy.items():
            if not m_info.get('active', True): 
                continue

            m_data = sim_data[m_key]
            if not m_data: 
                continue

            separate_ls = m_info.get("separate_ls", False)
            side = m_info.get("side", "both")
            if side == "both":
                side = ["long", "short"] if separate_ls else ["both"]

            for sd in side:
                series = m_data[sd]['data']
                returns = series.flatten()

                if len(returns) > (lookback * 0.5) and not np.all(returns == 0):
                    col_id = f"{m_key[0]}_{sd}"
                    matrix_data[col_id] = returns
                    model_map[col_id] = (m_key, side)

        # Not enough data
        if len(matrix_data) < 2:
            print(f"    < [PortfolioSystemManager._default_rebalance] lenght of matrix_data < 2")
            return hierarchy
        
        # Calculates HRP
        df_returns = pl.DataFrame(matrix_data).fill_null(0.0)
        try:
            hrp_engine = HRP()
            weights_dict = hrp_engine._calculate_logic(df_returns)
        except Exception as e:
            print(f"    < [PortfolioSystemManager._default_rebalance] HRP Failed in idx {i}: {e}")
            return hierarchy
        
        # Clears previous price
        for m_key, m_info in hierarchy['models'].items():
            for s in ["both", "long", "short"]:
                m_info['metrics'][s]['weight'] = 0.0

        # Applies new weights
        for col_id, weight in weights_dict.items():
            m_key, sd = model_map[col_id]
            hierarchy['models'][m_key][side]['weight'] = weight

        print(weights_dict)

        return hierarchy

    def _default_main(self, i, step_dt, hierarchy: dict, indicator_pool: dict, port_returns: dict, key) -> bool:
        
        # Default uses aggr of models for Portfolio Level
        sim_data = self.get_data(key=key, lookback=self.reb_lookback, data_type="aggr") 
        print(sim_data)

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


















