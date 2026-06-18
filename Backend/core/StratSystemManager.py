from dataclasses import dataclass
from SystemManager import SystemManager, SystemManagerParams
from typing import Literal, Dict, List
import polars as pl

@dataclass
class StratSystemManagerParams(SystemManagerParams):
    parset_order: Literal["highest", "lowest", "mode"] = "highest"
    parset_metric: Literal["pnl", "sharpe", "pnl_dd"] = "pnl"
    parset_allocation: Literal["1/n", "custom"] = "1/n"
    parset_number_cutoff: int = 1

    parset_sides_overwrite: str = None # Overwrite hierachy to selects only from one specific side
    close_open_trades_on_rebalance: bool = False
    wf_recreate_with_curr_ps_ids: bool = False # If True then recreates wf with only curr ps_ids

    # NOTE Importante colocar opção de habilitar entre os ps_ids disponíveis ou usar todos os que foram enviados (existe apenas rebalance nesse caso entre todos)
    # NOTE wf padrão olha todos os parsets em parquet e automaticamente ajusta (não precisa enviar ps_ids junto), wf rolling deve olhar os disponíveis e realizar o wf manualmente   

class StratSystemManager(SystemManager): 
    def __init__(self, ssm_params: SystemManagerParams):
        super().__init__(ssm_params) 
        self.parset_order = ssm_params.parset_order
        self.parset_metric = ssm_params.parset_metric
        self.parset_allocation = ssm_params.parset_allocation
        self.parset_number_cutoff = ssm_params.parset_number_cutoff
        self.parset_sides_overwrite = ssm_params.parset_sides_overwrite
        self.close_open_trades_on_rebalance = ssm_params.close_open_trades_on_rebalance
        self.wf_recreate_with_curr_ps_ids = ssm_params.wf_recreate_with_curr_ps_ids

#||=========================================================================================||

    def _default_pre_compute(self, global_assets, timeline, aggr_ret, indicator_pool, param_sets, manager_level_key) -> dict:
        # By Default doesn't calculate anything else, but can be used to prepare signals or other stuff != indicators
        return indicator_pool
                       
    def _default_rank(self, i, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key) -> Dict[str, float]:
        # Calculates performance metric for each parset in selected lookback
        strat_node = hierarchy.get(key)
        if not strat_node:
            return hierarchy
        
        scores = {}
        param_sets_dict = strat_node.get("param_sets", {})

        strat_side = strat_node.get("side", "BOTH").upper()

        valid_sides = ["long", "short"] if strat_side == "SEPR" else [strat_side.lower()]

        for ps_id in param_sets_dict.keys():
            for side_key in valid_sides:
                # Gets historical returns from specific param_sets
                virtual_ps_id = f"{ps_id}_{side_key}"
                df_ps = sim_data.get(side_key, {}).get(ps_id)

                if df_ps is not None and not df_ps.is_empty() and "pnl" in df_ps.columns:
                    if self.parset_metric == "pnl":
                        score_val = df_ps["pnl"].sum()
                    elif self.parset_metric == "sharpe":
                        mean_pnl = df_ps["pnl"].mean()
                        std_pnl = df_ps["pnl"].std()
                        score_val = (mean_pnl / std_pnl) if std_pnl and std_pnl > 0 else -999.0
                    elif self.parset_metric == "pnl_dd":
                        dd_df = df_ps.select([
                            pl.col("pnl"),
                            pl.col("pnl").cum_sum().alias("cum_pnl")
                        ]).select([
                            pl.col("pnl"),
                            (pl.col("cum_pnl").cum_max() - pl.col("cum_pnl")).alias("dd")
                        ])
                        max_dd = dd_df["dd"].max()
                        total_pnl = dd_df["pnl"].sum()
                        score_val = (total_pnl / max_dd) if max_dd and max_dd > 0 else total_pnl
                    else:
                        score_val = -999.0
                else:
                    score_val = -999.0
                scores[virtual_ps_id] = score_val

        strat_node["_scores"] = scores # Has score for each ps_id in this window
        return hierarchy

    def _default_filter(self, i, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key) -> List[str]:
        # Orders ParamSets by generated scores and applies cutoff
        strat_node = hierarchy.get(key)
        if not strat_node or "_scores" not in strat_node:
            return hierarchy
        
        scores = strat_node["_scores"]
        is_reverse = True if self.parset_order == "highest" else False
        
        sorted_setups = sorted(
            [item for item in scores.items() if item[1] != -999.0], 
            key=lambda x: x[1], 
            reverse=is_reverse
        )

        # Case of no valid setup
        if not sorted_setups:
            sorted_setups = sorted(scores.items(), key=lambda x: x[1], reverse=is_reverse)

        # Selects N best and cutoff
        cutoff_setups = [ps_id for ps_id, _ in sorted_setups[:self.parset_number_cutoff]]

        strat_node["_active_setups"] = cutoff_setups
        return hierarchy 

    def _default_rebalance(self, i, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key) -> List[str]:
        # Calculates weight distribution

        strat_node = hierarchy.get(key)
        if not strat_node or "_active_setups" not in strat_node:
            return hierarchy

        active_setups = strat_node["_active_setups"]
        if not active_setups:
            strat_node["weights"] = {}
            return hierarchy
        
        if self.parset_allocation == "1/n":
            target_weight = 1.0 / len(active_setups)
            strat_node["weights"] = {ps_id: target_weight for ps_id in active_setups}

        elif self.parset_allocation == "custom":
            scores = strat_node.get("_scores", {})
            valid_scores = {ps_id: max(0.0, scores.get(ps_id, 0.0)) for ps_id in active_setups}
            sum_scores = sum(valid_scores.values())
            
            if sum_scores > 0:
                strat_node["weights"] = {ps_id: score / sum_scores for ps_id, score in valid_scores.items()}
            else: # Fallback to 1/N if doesn't recognize
                target_weight = 1.0 / len(active_setups)
                strat_node["weights"] = {ps_id: target_weight for ps_id in active_setups}

        return hierarchy

    # ── Every Datetime [i] ───────────────────────────────────────────────

    def _default_main(self, i, step_dt, hierarchy: dict, indicator_pool: dict, port_returns: dict, key) -> bool:
        strat_node = hierarchy.get(key, {})
        strat_side = strat_node.get("side", "BOTH").upper()
        valid_sides = ["long", "short"] if strat_side == "SEPR" else [strat_side.lower()]
        sim_data = self.get_data(key=key, lookback=self.reb_lookback, data_type="aggr_dynamic", side=valid_sides)

        if not sim_data:
            print(f"    < [StratSystemManager._default_main] Warning, no sim_data for step {i}")
            return hierarchy

        hierarchy = self.rank(i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)
        hierarchy = self.filter(i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)
        hierarchy = self.rebalance(i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)

        return hierarchy

#||=========================================================================================||


















