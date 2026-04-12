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

        self._pre_cache: Dict = {}   # Metrics and Indicators
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
                       
    def _default_rank(self, step_dt, hierarchy, indicator_pool, op_data, port_returns) -> dict:
            """
            op_data: DataFrame Largo [datetime, Modelo_A, Modelo_B, ...]
            hierarchy: dicionário contendo a lista de modelos ativos em hierarchy["models"]
            """
            models_to_rank = hierarchy.get("models", [])
            
            # Se não houver dados ou modelos para rankear, retorna a hierarquia original
            if not models_to_rank or op_data is None:
                return hierarchy

            # 1. Extraímos a última linha (estado atual) como um dicionário
            # { "MA Trend Following": [0.00883], "Outro Modelo": [0.005] }
            last_values = op_data.tail(1).to_dict(as_series=False)
            
            # 2. Define a direção da ordenação (highest = descendente)
            descending = self.model_hierarchy.get("order_by", "highest") == "highest"

            # 3. Ordenação baseada nos valores das colunas correspondentes
            # O str(m) garante que comparamos o nome do modelo com o nome da coluna no DF
            ranked = sorted(
                models_to_rank, 
                key=lambda m: last_values.get(str(m), [-float("inf")])[0], 
                reverse=descending
            )

            # 4. Atualizamos a lista na hierarquia
            hierarchy["models"] = ranked
            
            return hierarchy

    def _default_filter(self, step_dt, hierarchy, indicator_pool, op_data, port_returns) -> List[str]:

        for m_name, m_obj in op_data.items():
            #var = self.get_ind("var", m_name, step_dt, indicator_pool, ) # NOTE COMO USUÁRIO VAI SABER O PS_KEY? NÃO DEVERIA PRECISAR, APENAS PUXAR PARA O NIVEL ATUAL AUTOMATICO
            if hierarchy["models"][m_name]["pnl"] < 0.4: hierarchy["models"][m_name]["pnl"] = -float("inf") 
        return hierarchy # By default doesn't filter out any model

    def _default_rebalance(self, step_dt, hierarchy, indicator_pool, op_data, port_returns) -> List[str]:
        models = hierarchy["models"]

        descending = self.model_hierarchy.get("order_by", "highest") == "highest"
        ranked = sorted(models, key=lambda m: models['pnl'].get(m, -float("inf")), reverse=descending)

        if self.max_active_models:
            hierarchy["models"] = ranked[:self.max_active_models]

        return hierarchy

    # ── Every Datetime [i] ───────────────────────────────────────────────
    
    def _default_main(self, step_dt, hierarchy: dict, indicator_pool: dict, op_data: dict, port_returns: dict) -> bool:
        print(op_data)
        hierarchy = self.rank(step_dt, hierarchy, indicator_pool, op_data, port_returns)
        hierarchy = self.filter(step_dt, hierarchy, indicator_pool, op_data, port_returns)
        hierarchy = self.rebalance(step_dt, hierarchy, indicator_pool, op_data, port_returns)

        return hierarchy
   
#||=========================================================================================||

    """ Dt execution framework

    1. Check current tradable Models
    -> REBALANCE

    2. New Rank generated with updated data
    3. Needs to remove any Models? if yes then close or keep positions open by SM rules? != MM rules
    4. Needs to add any Models? 
    

    """


















