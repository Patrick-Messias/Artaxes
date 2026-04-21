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

    def _default_pre_compute(self, global_assets, timeline, aggr_ret, indicator_pool, param_sets) -> dict:
        # By Default doesn't calculate anything else, but can be used to prepare signals or other stuff != indicators
        return indicator_pool
                       
    # ── Every Datetime [i] ───────────────────────────────────────────────
    
    def _default_rank(self, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key) -> dict:
        # Rankea os modelos baseados na performance contida no DataFrame sim_data.
        # No nível de Portfólio, os 'filhos' são modelos.
        # Tentamos buscar 'models', se não existir, buscamos 'strats' (para reuso da lógica)
        entities = hierarchy.get("models", [])
        # if not entities or sim_data is None:
        #     print("     < [PortofolioSystemManager._default_rank] No entities to rank or sim_data is None. Skipping ranking.")
        #     return hierarchy, indicator_pool, sim_data, port_returns

        #ARRUMAR get_ind DEVE AJUSTAR A KEY AUTOMATICAMENTE E PUXAR POR NOME

        # Data 
        vol = self.get_ind(ind_key=key, target="vol", i=step_dt, ps_name=None) 
        print(vol)
        # NOTE Modificar para ter acesso ao ind com e sem opção de tuple key
        #e/ou salvar com identificador de tuple (op, m, s, a)

        # Performance Calc
        performance = {}
        for entity in entities:
            col_name = str(entity)
            if col_name in sim_data.columns:
                # Soma do PnL acumulado no período de lookback
                performance[entity] = sim_data.get_column(col_name).sum()
            else:
                performance[entity] = -float("inf")

        # Ranking
        descending = self.model_hierarchy.get("order_by", "highest") == "highest"
        ranked = sorted(
            entities, 
            key=lambda e: performance.get(e, -float("inf")), 
            reverse=descending
        )

        hierarchy["models"] = ranked
        return hierarchy, indicator_pool, sim_data, port_returns

    def _default_filter(self, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key) -> dict:
        """
        Filtra entidades. Por padrão, mantém todas. 
        Pode ser expandido para remover modelos com drawdown excessivo usando o sim_data.
        """
        return hierarchy, indicator_pool, sim_data, port_returns 

    def _default_rebalance(self, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key) -> dict:
        entities = hierarchy.get("models", [])
        if not entities:
            return hierarchy, indicator_pool, sim_data, port_returns

        # 1. Aplicar o corte (Slicing) - Se max_active for 3, pegamos os 3 melhores do ranking
        active_entities = entities
        if self.max_active_models is not None:
            active_entities = entities[:self.max_active_models]
        
        # 2. Distribuição 1/n (Equal Weight)
        # Criamos um mapa de pesos onde quem não está no topo recebe peso 0
        num_active = len(active_entities)
        weight_per_entity = 1.0 / num_active if num_active > 0 else 0.0
        
        # Guardamos isso na hierarchy para o PMM (Money Manager) ler depois
        hierarchy["weights"] = {entity: weight_per_entity for entity in active_entities}
        
        # Atualizamos a lista de ativos para que apenas os selecionados processem ordens
        hierarchy["models"] = active_entities

        return hierarchy, indicator_pool, sim_data, port_returns

    def _default_main(self, step_dt, hierarchy: dict, indicator_pool: dict, port_returns: dict, key) -> bool:
        
        # Default uses aggr of models for Portfolio Level
        sim_data = self.get_data(key=key, lookback=self.reb_lookback, data_type="aggr", side="both") # NOTE MUST BE PORTF_AGGR NOT OPERA_AGGR NOTE # 

        hierarchy, indicator_pool, sim_data, port_returns  = self.rank(step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)
        hierarchy, indicator_pool, sim_data, port_returns  = self.filter(step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)
        hierarchy, indicator_pool, sim_data, port_returns  = self.rebalance(step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)

        return hierarchy
   
#||=========================================================================================||

    """ Dt execution framework

    1. Check current tradable Models
    -> REBALANCE

    2. New Rank generated with updated data
    3. Needs to remove any Models? if yes then close or keep positions open by SM rules? != MM rules
    4. Needs to add any Models? 
    

    """


















