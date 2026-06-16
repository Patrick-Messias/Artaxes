from dataclasses import dataclass, field
from SystemManager import SystemManager, SystemManagerParams
from typing import Optional, Callable, Dict, List
import polars as pl, numpy as np

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

    def _default_pre_compute(self, global_assets, timeline, aggr_ret, indicator_pool, param_sets, manager_level_key) -> dict:
        # By Default doesn't calculate anything else, but can be used to prepare signals or other stuff != indicators
        return indicator_pool
                       
    def _default_rank(self, i, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key) -> Dict[str, float]:
        df_rets = pl.DataFrame(sim_data.get('both', {})).fill_null(0.0)
        if df_rets.is_empty() or df_rets.width < 1: return {}

        scores = {}
        corr_matrix = df_rets.corr() if df_rets.width > 1 else None

        for col in df_rets.columns:
            series = df_rets[col]
            std = series.std()
            sharpe = (series.mean() / std * np.sqrt(252)) if std > 0 else 0.0

            if corr_matrix is not None:
                avg_corr = (corr_matrix[col].sum() - 1) / (df_rets.width - 1)
            else:
                avg_corr = 0.0

            scores[col] = sharpe * (1 - avg_corr)
        
        return scores

    def _default_filter(self, i, step_dt, hierarchy: dict, indicator_pool: dict, scores: dict, port_returns: dict, key) -> List[str]:
        # Enables only top N asset/strat based on ranking
        top_n = getattr(self.params, 'top_n', 5)
        max_asset_per_strat_n = getattr(self.params, 'max_asset_per_strat_n', None)
        order_by = self.model_hierarchy['order_by']

        # Filters only those with score > 0 and takes N self.model_hierarchy['order_by']
        valid_scores = {k: v for k, v in scores.items() if v > 0}
        ranked_keys = sorted(valid_scores, key=valid_scores.get, reverse=True)[:top_n]

        for s_name, s_node in hierarchy.get('strats', {}).items():
            for a_name, a_node in s_node.get('assets', {}).items():
                item_key = f"{s_name}_{a_name}"

                if item_key in ranked_keys:
                    a_node['active'] = True
                    a_node['score'] = scores[item_key]
                else:
                    a_node['active'] = False
                    a_node['score'] = 0.0

        return hierarchy
    
    def _generate_internal_weights(self, i, step_dt, hierarchy: dict, scores: dict) -> dict:
        # Converts approved asset scores in percent weights that sum up to 1.0, later MM will use this to apply capital
        total_score = 0.0
        active_nodes = []

        for s_name, s_node in hierarchy.get("strats", {}).items():
            for a_name, a_node in s_node.get('assets', {}).items():
                if a_node.get('active', False):
                    total_score += a_node.get('score', 0.0)
                    active_nodes.append(a_node)

        # Distribute weigts proportionally
        for a_node in active_nodes:
            if total_score > 0:
                relative_weight = a_node.get('score', 0.0) / total_score
            else:
                relative_weight = 1.0 / len(active_nodes)

            # Applies weights on hierarchy
            l_factor = getattr(self.params, 'long_factor', 0.5)
            s_factor = getattr(self.params, 'short_factor', 0.5)

            if 'long' in a_node: a_node['long']['weight'] = relative_weight * l_factor 
            if 'short' in a_node: a_node['short']['weight'] = relative_weight * s_factor
            if 'both' in a_node: a_node['both']['weight'] = relative_weight

        return hierarchy
    

    def _default_rebalance(self, i, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key) -> List[str]:
        return hierarchy

    # ── Every Datetime [i] ───────────────────────────────────────────────
    
    def _default_main(self, i, step_dt, hierarchy: dict, indicator_pool: dict, port_returns: dict, key) -> dict:
        lookback = getattr(self.params, 'reb_lookback', 63)
        #sim_data = self.get_data(key=key, lookback=lookback, data_type="aggr", side="both")

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


