from dataclasses import dataclass, field
from SystemManager import SystemManager, SystemManagerParams
from typing import List, Optional, Dict, Literal, Callable, Union
#from Backend.core import Asset
import polars as pl, numpy as np

@dataclass
class PortfolioSystemManagerParams(SystemManagerParams):
    parset_order: Literal["highest", "lowest", "mode"] = "highest"
    parset_metric: Literal["pnl", "sharpe", "pnl_dd"] = "pnl"
    parset_allocation: Literal["1/n", "pnl", "HRP"] = "1/n"
    parset_number_cutoff: int = 1
    parset_sides_overwrite: str = None

class PortfolioSystemManager(SystemManager): # Manages portfolio's model hierarchy 
    def __init__(self, psm_params: PortfolioSystemManagerParams):
        super().__init__(psm_params)
        self.parset_order = psm_params.parset_order
        self.parset_metric = psm_params.parset_metric
        self.parset_allocation = psm_params.parset_allocation
        self.parset_number_cutoff = psm_params.parset_number_cutoff

#||=========================================================================================||

    def _default_pre_compute(self, global_assets, timeline, aggr_ret, indicator_pool, param_sets, manager_level_key) -> dict:
        # By Default doesn't calculate anything else, but can be used to prepare signals or other stuff != indicators
        return indicator_pool
                       
    # ── Every Datetime [i] ───────────────────────────────────────────────
    
    def _default_rank(self, i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key) -> dict:
        series_dict = {}

        # Constructs virtual model entities
        for m_name, m_node in hierarchy.get('models', {}).items():
            model_side = m_node.get('side', 'BOTH').upper()
            
            # # Dismembers into two different entities 
            valid_sides = ['long', 'short'] if model_side == 'SEPR' else [model_side.lower()]
            
            for side_key in valid_sides:
                virtual_m_id = f"{m_name}_{side_key}"
                
                # Assumes that sim_data "aggr" has data aggregated by sides
                if side_key in sim_data and m_name in sim_data[side_key]:
                    series_dict[virtual_m_id] = sim_data[side_key][m_name]

        df_rets = pl.DataFrame(series_dict).fill_null(0.0)
        scores = {}

        # Calculates metrics for each model virtual entity
        if not df_rets.is_empty() and df_rets.width > 0:
            for col in df_rets.columns:
                series = df_rets[col]

                if self.parset_metric.get("metric") == "pnl":
                    score_val = series.sum()
                    
                elif self.parset_metric.get("metric") == "sharpe":
                    std = series.std()
                    score_val = (series.mean() / std * np.sqrt(252)) if std and std > 0 else -999.0
                    
                elif self.parset_metric.get("metric") == "pnl_dd":
                    dd_df = df_rets.select([
                        pl.col(col).alias("ret"),
                        pl.col(col).cum_sum().alias("cum_pnl")
                    ]).select([
                        pl.col("ret"),
                        (pl.col("cum_pnl").cum_max() - pl.col("cum_pnl")).alias("dd")
                    ])
                    max_dd = dd_df["dd"].max()
                    total_pnl = dd_df["ret"].sum()
                    score_val = (total_pnl / max_dd) if max_dd and max_dd > 0 else total_pnl
                else:
                    score_val = -999.0
                scores[col] = score_val

        hierarchy["_scores"] = scores
        return hierarchy

    def _default_filter(self, i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key) -> dict:
        # Filters and selects top N models
        scores = hierarchy.get("_scores", {})

        # Removes invalids
        valid_scores = {k: v for k, v in scores.items() if v > -999.0}
        
        is_reverse = True if self.parset_order == "highest" else False
        ranked_keys = sorted(valid_scores, key=valid_scores.get, reverse=is_reverse)

        if self.parset_number_cutoff is not None:
            ranked_keys = ranked_keys[:self.parset_number_cutoff]

        hierarchy["_active_models"] = ranked_keys
        return hierarchy

    def _default_rebalance(self, i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key) -> dict:
        import numpy as np, polars as pl
        from indicators.HRP import HRP
        
        active_models = hierarchy.get("_active_models", [])
        scores = hierarchy.get("_scores", {})
        weights_dict = {}

        if not active_models:
            hierarchy["weights"] = weights_dict
            return hierarchy

        if self.parset_allocation == "1/N":
            target_weight = 1.0 / len(active_models)
            weights_dict = {m_id: target_weight for m_id in active_models}
            
        elif self.parset_allocation == "pnl":
            valid_scores = {m_id: max(0.0, scores.get(m_id, 0.0)) for m_id in active_models}
            total_score = sum(valid_scores.values())
            
            if total_score > 0:
                weights_dict = {m_id: score / total_score for m_id, score in valid_scores.items()}
            else:
                target_weight = 1.0 / len(active_models)
                weights_dict = {m_id: target_weight for m_id in active_models}
                
        elif self.parset_allocation == "risk_parity":
            matrix_data = {}
            for m_id in active_models:
                parts = m_id.rsplit('_', 1)
                if len(parts) != 2: continue

                m_name, side_key = parts[0], parts[1]

                if side_key in sim_data and m_name in sim_data[side_key]:
                    raw_returns = np.array(sim_data[side_key][m_name], dtype=np.float64)
                    if len(raw_returns) > 1 and not np.all(np.isnan(raw_returns)):
                        matrix_data[m_id] = np.nan_to_num(raw_returns, nan=0.0, posinf=0.0)

            # If not enough models for corr matrix, makes fallback to Score
            if len(matrix_data) < 2:
                valid_scores = {m_id: max(0.0, scores.get(m_id, 0.0)) for m_id in active_models}
                total_score = sum(valid_scores.values())
                weights_dict = {m_id: score / total_score for m_id, score in valid_scores.items()} if total_score > 0 else {m_id: 1.0 / len(active_models) for m_id in active_models}
                hierarchy["weights"] = weights_dict
                return hierarchy
            
            try:
                explicit_schema = {name: pl.Float64 for name in matrix_data.keys()}
                df_hrp = pl.DataFrame(matrix_data, schema=explicit_schema).fill_null(0.0).fill_nan(0.0)
                
                hrp_engine = HRP()
                hrp_weights = hrp_engine._calculate_logic(df_hrp)

                # Distribution and filtering
                final_active = []
                for m_id, weight in hrp_weights.items():
                    w_float = float(weight)
                    if w_float > 0.001:  # Filtra ativos que receberam menos de 0.1% de peso pelo HRP
                        weights_dict[m_id] = w_float
                        final_active.append(m_id)
                hierarchy["_active_models"] = final_active

                # Normalizes ramaining weight to sum up to 1.0
                total_w = sum(weights_dict.values())
                if total_w > 0:
                    hierarchy["weights"] = {k: v / total_w for k, v in weights_dict.items()}
                else:
                    hierarchy["weights"] = {}

            except Exception as e:
                print(f"    < [PortfolioSystemManager._default_rebalance] HRP failed at idx {i} resorting to fallback 1/N: {e}")
                # Fallback in case of math error
                hierarchy["weights"] = {m_id: 1.0 / len(active_models) for m_id in active_models}

        # {"Model1_long": 0.33, "Model1_short": 0.33, "Model3_both": 0.34}
        hierarchy["weights"] = weights_dict
        return hierarchy


    def _default_main(self, i, step_dt, hierarchy: dict, indicator_pool: dict, port_returns: dict, key) -> bool:
        
        # Default uses aggr of models for Portfolio Level
        port_key = (self.portfolio.name,)
        port_aggr_data = self.get_data(key=port_key, lookback=self.reb_lookback, data_type="aggr", side=["both", "long", "short"])

        if not port_aggr_data:
            print("     < [PSM._default_main] No aggr data found")
            return hierarchy

        # 2. Roda o Rank (Vetorizado sobre a matriz global)
        hierarchy = self._default_rank(i, step_dt, hierarchy, indicator_pool, port_aggr_data, port_returns, port_key)

        # 3. Roda o Filter (Decide quem fica ativo baseado nos scores)
        hierarchy = self._default_filter(i, step_dt, hierarchy, indicator_pool, port_aggr_data, port_returns, port_key)

        # 4. Roda o Rebalance (Calcula pesos para os que sobreviveram)
        hierarchy = self._default_rebalance(i, step_dt, hierarchy, indicator_pool, port_aggr_data, port_returns, port_key)

        return hierarchy
   
#||=========================================================================================||

    """ Dt execution framework

    1. Check current tradable Models
    -> REBALANCE

    2. New Rank generated with updated data
    3. Needs to remove any Models? if yes then close or keep positions open by SM rules? != MM rules
    4. Needs to add any Models? 
    

    """


















