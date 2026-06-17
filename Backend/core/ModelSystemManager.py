from dataclasses import dataclass, field
from SystemManager import SystemManager, SystemManagerParams
from typing import Literal, Dict, List
import polars as pl, numpy as np

@dataclass
class ModelSystemManagerParams(SystemManagerParams):
    parset_order: Literal["highest", "lowest", "mode"] = "highest"
    parset_metric: Literal["pnl", "sharpe", "pnl_dd"] = "pnl"
    parset_allocation: Literal["1/n", "custom"] = "1/n"
    parset_number_cutoff: int = 1
    parset_sides_overwrite: str = None

class ModelSystemManager(SystemManager): # Manages portfolio's model hierarchy 
    def __init__(self, msm_params: ModelSystemManagerParams):
        super().__init__(msm_params) # SystemManager attributes init
        self.parset_order = msm_params.parset_order
        self.parset_metric = msm_params.parset_metric
        self.parset_allocation = msm_params.parset_allocation
        self.parset_number_cutoff = msm_params.parset_number_cutoff
        self.parset_sides_overwrite = msm_params.parset_sides_overwrite

#||=========================================================================================||

    def _default_pre_compute(self, global_assets, timeline, aggr_ret, indicator_pool, param_sets, manager_level_key) -> dict:
        # By Default doesn't calculate anything else, but can be used to prepare signals or other stuff != indicators
        return indicator_pool
                       
    def _default_rank(self, i, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key) -> Dict[str, float]:
        model_node = hierarchy.get(key)
        if not model_node:
            return hierarchy
        series_dict = {}

        for s_name, s_node in model_node.get("strats", {}).items():
            # "BOTH", "LONG", "SHORT", "SEPR"
            strat_side = s_node.get('side', 'BOTH').upper()

            for a_name in s_node.get('assets', {}).keys():
                col_name = f"{s_name}_{a_name}"

                if strat_side in ['BOTH', 'LONG', 'SHORT']:
                    side_key = strat_side.lower()
                    if side_key in sim_data and col_name in sim_data[side_key]:
                        # Creates unique entity
                        series_dict[f"{col_name}_{side_key}"] = sim_data[side_key][col_name]
                elif strat_side == "SEPR": # Dismembers into two different entities
                    if "long" in sim_data and col_name in sim_data["long"]:
                        series_dict[f"{col_name}_long"] = sim_data['long'][col_name]
                    if 'short' in sim_data and col_name in sim_data['short']:
                        series_dict[f"{col_name}_short"] = sim_data['short'][col_name]

        df_rets = pl.DataFrame(series_dict).fill_null(0.0)
        scores = {}

        # Calculates Sharpe and Corr for entities
        if not df_rets.is_empty() and df_rets.width > 0:
            corr_matrix = df_rets.corr() if df_rets.width > 1 else None

            for col in df_rets.columns:
                series = df_rets[col]
                std = series.std()

                if std and std>0:
                    sharpe = (series.mean() / std / np.sqrt(252))
                else:
                    sharpe = -999.0

                if corr_matrix is not None and sharpe > -999.0:
                    avg_corr = (corr_matrix[col].sum()-1 / (df_rets.width-1))
                    score_val = sharpe * (1-avg_corr)
                else:
                    score_val = sharpe
                
                scores[col] = score_val
        
        model_node["_scores"] = scores
        return hierarchy

    def _default_filter(self, i, step_dt, hierarchy: dict, indicator_pool: dict, scores: dict, port_returns: dict, key) -> List[str]:
        model_node = hierarchy.get(key)
        if not model_node or "_scores" not in model_node:
            return hierarchy
        
        scores = model_node["_scores"]
        valid_scores = {k: v for k, v in scores.items() if v > -999.0}
        is_reverse = True if self.model_hierarchy.get("order_by", "highest") == "highest" else False

        ranked_keys = sorted(valid_scores, key=valid_scores.get, reverse=is_reverse)[:self.parset_number_cutoff]

        model_node["_active_combos"] = ranked_keys
        return hierarchy    

    def _default_rebalance(self, i, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key) -> List[str]:
        model_node = hierarchy.get(key)
        if not model_node or "_active_combos" not in model_node:
            return hierarchy

        active_combos = model_node["_active_combos"]
        scores = model_node.get("_scores", {})
        weights_dict = {}

        if not active_combos:
            model_node["weights"] = weights_dict
            return hierarchy

        if self.parset_allocation == "equal":
            target_weight = 1.0 / len(active_combos)
            weights_dict = {combo: target_weight for combo in active_combos}
            
        elif self.parset_allocation == "performance_weighted":
            valid_scores = {combo: max(0.0, scores.get(combo, 0.0)) for combo in active_combos}
            total_score = sum(valid_scores.values())
            
            if total_score > 0:
                weights_dict = {combo: score / total_score for combo, score in valid_scores.items()}
            else:
                target_weight = 1.0 / len(active_combos)
                weights_dict = {combo: target_weight for combo in active_combos}

        # Strats weights ex: "AT2_WIN$_long": 0.3, "AT2_WIN$_short": 0.15, "AT2_WIN$_both": 0.45
        model_node["weights"] = weights_dict
        
        return hierarchy

    # ── Every Datetime [i] ───────────────────────────────────────────────
    
    def _default_main(self, i, step_dt, hierarchy: dict, indicator_pool: dict, port_returns: dict, key) -> dict:
        lookback = getattr(self.params, 'reb_lookback', 63)
        print(key)
        sim_data = self.get_data(key=key, lookback=self.reb_lookback, data_type="aggr", side="both")

        if not sim_data:
            return hierarchy

        hierarchy = self.rank(i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)
        hierarchy = self.filter(i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)
        hierarchy = self.rebalance(i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)

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


