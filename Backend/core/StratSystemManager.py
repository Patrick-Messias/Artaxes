from dataclasses import dataclass, field
from SystemManager import SystemManager, SystemManagerParams
from typing import Literal, Dict, List

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

        
        return hierarchy

    def _default_filter(self, i, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key) -> List[str]:
        return hierarchy # By default doesn't filter out any model

    def _default_rebalance(self, i, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key) -> List[str]:
        return hierarchy

    # ── Every Datetime [i] ───────────────────────────────────────────────

    def _default_main(self, i, step_dt, hierarchy: dict, indicator_pool: dict, port_returns: dict, key) -> bool:

        sim_data = self.get_data(key=key, lookback=self.reb_lookback, data_type="aggr", side="both")

        hierarchy = self.rank(i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)
        hierarchy = self.filter(i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)
        hierarchy = self.rebalance(i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)

        return hierarchy

#||=========================================================================================||


















