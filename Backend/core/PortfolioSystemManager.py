from dataclasses import dataclass, field
from SystemManager import SystemManager, SystemManagerParams
from typing import List, Optional, Dict, Literal, Callable, Union
#from Backend.core import Asset
import polars as pl

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

        #self._pre_cache: Dict = {}   # Metrics and Indicators
        # self._pre_cache = {
        #     "models": {
        #         "Model_A": {
        #             "metrics": {"sharpe": [...], "pnl_dd": [...]}, # Alinhado à timeline
        #             "indicators": {"rsi_equity": [...]}
        #         }},
        #     "strats": { ... }}

#||=========================================================================================||

    def _default_pre_compute(self, global_assets, timeline, sim_data, aggr_ret, indicator_pool, param_sets) -> dict:
        # By Default doesn't calculate anything else, but can be used to prepare signals or other stuff != indicators
        return indicator_pool, sim_data
                       
    # ── Every Datetime [i] ───────────────────────────────────────────────
    
    def _default_rank(self, step_dt, hierarchy, indicator_pool, sim_data, port_returns) -> dict:
        # Rankea os modelos baseados na performance contida no DataFrame sim_data.
        # No nível de Portfólio, os 'filhos' são modelos.
        # Tentamos buscar 'models', se não existir, buscamos 'strats' (para reuso da lógica)
        children_key = "models" if "models" in hierarchy else "strats"
        entities_to_rank = hierarchy.get(children_key, [])

        if not entities_to_rank or sim_data is None:
            return hierarchy, indicator_pool, sim_data, port_returns

        # 1. Extraímos o estado atual (última linha)
        last_values = sim_data.tail(1).to_dict(as_series=False)
        
        # 2. Direção da ordenação
        descending = self.model_hierarchy.get("order_by", "highest") == "highest"

        # 3. Ordenação segura: usamos str(e) para bater com o nome da coluna no Polars
        ranked = sorted(
            entities_to_rank, 
            key=lambda e: last_values.get(str(e), [-float("inf")])[0], 
            reverse=descending
        )

        hierarchy[children_key] = ranked
        return hierarchy, indicator_pool, sim_data, port_returns

    def _default_filter(self, step_dt, hierarchy, indicator_pool, sim_data, port_returns) -> dict:
        """
        Filtra entidades. Por padrão, mantém todas. 
        Pode ser expandido para remover modelos com drawdown excessivo usando o sim_data.
        """
        return hierarchy, indicator_pool, sim_data, port_returns 

    def _default_rebalance(self, step_dt, hierarchy, indicator_pool, sim_data, port_returns) -> dict:
        """
        Aplica o limite de modelos ativos (max_active_models) após o ranking.
        """
        children_key = "models" if "models" in hierarchy else "strats"
        entities = hierarchy.get(children_key, [])

        if not entities:
            return hierarchy, indicator_pool, sim_data, port_returns

        # O ranking já foi feito no _default_rank (chamado pelo main antes do rebalance)
        # Aqui apenas aplicamos o corte de quantidade (Slicing)
        if self.max_active_models is not None:
            hierarchy[children_key] = entities[:self.max_active_models]

        return hierarchy, indicator_pool, sim_data, port_returns

    def _default_main(self, step_dt, hierarchy: dict, indicator_pool: dict, port_returns: dict, key: str) -> bool:
        
        # Default uses aggr of models for Portfolio Level
        sim_data = self.get_data(key=key, lookback=self.reb_lookback, data_type="aggr", side="BOTH")

        hierarchy, indicator_pool, sim_data, port_returns  = self.rank(step_dt, hierarchy, indicator_pool, sim_data, port_returns)
        hierarchy, indicator_pool, sim_data, port_returns  = self.filter(step_dt, hierarchy, indicator_pool, sim_data, port_returns)
        hierarchy, indicator_pool, sim_data, port_returns  = self.rebalance(step_dt, hierarchy, indicator_pool, sim_data, port_returns)

        return hierarchy
   
#||=========================================================================================||

    """ Dt execution framework

    1. Check current tradable Models
    -> REBALANCE

    2. New Rank generated with updated data
    3. Needs to remove any Models? if yes then close or keep positions open by SM rules? != MM rules
    4. Needs to add any Models? 
    

    """


















