"""
# System Management Algorithm (SMA) - Base class for all System Management
Função: orquestrar o comportamento do sistema em tempo de execução.
Liga/desliga Strats ou Models conforme regras globais.
Define quais combinações (Model + Asset + Strat) estão ativas.
Pode implementar lógica de auto-adaptação (ex: desativar modelos com drawdown alto).
Atua sobre os níveis superiores (controla quem “fala” com o PMM e o TM).
"""

import polars as pl, uuid
from typing import Literal, Set, Dict, Optional, Callable, List
from dataclasses import dataclass, field
from Indicator import Indicator
from BaseClass import BaseClass, BaseManager

@dataclass
class SystemManagerParams:
    name: str = field(default_factory=lambda: f'sm_{uuid.uuid4()}')

    reb_frequency: Literal["tick", "daily", "weekly", "monthly", "yearly", "never"] = "weekly"
    reb_lookback: int=252 # If len < lookback then [:idx]
    reb_lookback_period_type: Literal["tick", "day", "week", "month", "year"]="day" # 252 what? ticks, days?
    
    # Dados externos para o System Manager (Ex: Calendário Econômico, Sentimento, CDT)
    # Migrado para usar dicionário de Polars DataFrames
    assets: Set[str] = field(default_factory=set)

    # Customizable parameters for specific System Managers (Ex: thresholds para desativar modelos, regras de ativação, etc)
    params: Dict = field(default_factory=dict) 
    
    # Indicadores administrativos (Ex: Medidores de Regime de Mercado)
    indicators: Optional[Dict[str, Indicator]] = field(default_factory=dict)
    
    # Plugin functions for custom model hierarchy rules and rebalancing logic
    fn_pre_compute:     Optional[Callable] = None   # (history: Dict[str, pl.DataFrame]) -> None
    fn_rank:            Optional[Callable] = None   # (context: dict) -> Dict[str, float]
    fn_filter:          Optional[Callable] = None   # (context: dict) -> List[str]
    fn_rebalance:       Optional[Callable] = None   # (context: dict) -> List[str]
    fn_main:            Optional[Callable] = None   # (model_name: str, context: dict) -> bool

class SystemManager(BaseClass, BaseManager): 
    def __init__(self, sm_params: SystemManagerParams):
        super().__init__()
        self.name = sm_params.name
        self.reb_frequency = sm_params.reb_frequency
        self.reb_lookback = sm_params.reb_lookback
        self.reb_lookback_period_type = sm_params.reb_lookback_period_type
    
        # Custom Data & Rules
        self.assets = sm_params.assets
        self.params = sm_params.params
        self.indicators = sm_params.indicators

        # Funções plugáveis — usa custom se passado, senão usa default interno
        self._fn_pre_compute    = sm_params.fn_pre_compute
        self._fn_rank           = sm_params.fn_rank
        self._fn_filter         = sm_params.fn_filter
        self._fn_rebalance      = sm_params.fn_rebalance
        self._fn_main           = sm_params.fn_main

        self.portfolio = None

#||=========================================================================================||

    # ── Every Datetime [i] ───────────────────────────────────────────────

    def rank(self, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key) -> Dict[str, float]:
        # Ranks each model by metric defined in model_hierarchy. Returns dict[model_name: score]
        return self._call(self._fn_rank, self._default_rank, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)

    def filter(self, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key) -> List[str]:
        # Removes models that don't pass the filter function
        # Returns list of model_names that are active
        return self._call(self._fn_filter, self._default_filter, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)

    def rebalance(self, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key) -> List[str]:
        # Orchestrates rank -> filter -> selection
        # Returns ordered list of active models
        return self._call(self._fn_rebalance, self._default_rebalance, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)

    def main(self, step_dt, hierarchy: dict, indicator_pool: dict, port_returns: dict, key) -> bool:
        # Called every datetime for each model and asset
        # Returns True if model can operate now
        return self._call(self._fn_main, self._default_main, step_dt, hierarchy, indicator_pool, port_returns, key)

    # def __repr__(self):
    #     return f"<{self.__class__.__name__} name={self.name}>"

#||=========================================================================================||

    # Management Indicators
    def fama_french(): pass # Imports all T-Bills, Assets, etc
        
    def mae_mpe(): pass # 
    























