"""
# Money Management Algorithm (SMM / MMM / PMM) - Base class for all Money Management
Função: controlar risco, exposição e alocação de capital.
Camadas:
SMM (Strategy Money Management): define quanto alocar por trade dentro da estratégia.
MMM (Model Money Management): define quanto cada estratégia do modelo recebe.
PMM (Portfolio Money Management): define quanto cada modelo recebe do portfólio.
"""

import polars as pl
import uuid
from typing import Literal, Dict, Optional, Callable, List
from dataclasses import dataclass, field
from Indicator import Indicator
from BaseClass import BaseClass, BaseManager

@dataclass
class MoneyManagerParams:
    name: str = field(default_factory=lambda: f'mm_{uuid.uuid4()}')

    capital: float=100000.0
    max_capital_exposure: float=1.0
    
    reb_frequency: Literal["tick", "daily", "weekly", "monthly", "yearly", "never"] = "weekly"
    reb_lookback: int=252 # If len < lookback then [:idx]
    reb_lookback_period_type: Literal["tick", "day", "week", "month", "year"]="day" # 252 what? ticks, days?
 
    # Dados externos para MM (Ex: volatilidade do mercado, regime de juros)
    # Agora usa Polars DataFrame
    assets: Dict[str, pl.DataFrame] = field(default_factory=dict)

    # Customizable parameters for specific System Managers (Ex: thresholds para desativar modelos, regras de ativação, etc)
    params: Dict = field(default_factory=dict) 
    
    # Indicadores específicos para balanceamento de ativos/modelos
    indicators: Optional[Dict[str, Indicator]] = field(default_factory=dict) 

    # Plugin functions for custom model hierarchy rules and rebalancing logic
    fn_pre_compute:     Optional[Callable] = None   # (history: Dict[str, pl.DataFrame]) -> None
    fn_allocate:        Optional[Callable] = None   # (context: dict) -> Dict[str, float]
    fn_size:            Optional[Callable] = None   # (context: dict) -> List[str]
    fn_risk_guard:      Optional[Callable] = None   # (context: dict) -> List[str]
    fn_main:            Optional[Callable] = None   # (model_name: str, context: dict) -> bool

class MoneyManager(BaseClass, BaseManager): # Classe base para SMM, MMM e PMM
    def __init__(self, mm_params: MoneyManagerParams):
        super().__init__()
        self.name = mm_params.name
        self.reb_frequency = mm_params.reb_frequency
        self.reb_lookback = mm_params.reb_lookback
        self.reb_lookback_period_type = mm_params.reb_lookback_period_type
        
        # Custom Rules & Data
        self.assets = mm_params.assets
        self.params = mm_params.params
        self.indicators = mm_params.indicators

        # Funções plugáveis — usa custom se passado, senão usa default interno
        self._fn_pre_compute    = mm_params.fn_pre_compute
        self._fn_allocate       = mm_params.fn_allocate
        self._fn_size           = mm_params.fn_size
        self._fn_risk_guard     = mm_params.fn_risk_guard
        self._fn_main           = mm_params.fn_main

        self.portfolio = None

#||=========================================================================================||

    # ── Every Datetime [i] ───────────────────────────────────────────────

    def allocate(self, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key) -> Dict[str, float]:
        # Ranks each model by metric defined in model_hierarchy. Returns dict[model_name: score]
        return self._call(self._fn_allocate, self._default_allocate, step_dt, hierarchy, indicator_pool, sim_data, port_returns)

    def size(self, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key) -> List[str]:
        # Removes models that don't pass the filter function
        # Returns list of model_names that are active
        return self._call(self._fn_size, self._default_size, step_dt, hierarchy, indicator_pool, sim_data, port_returns)

    def risk_guard(self, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key) -> List[str]:
        # Orchestrates rank -> filter -> selection
        # Returns ordered list of active models
        return self._call(self._fn_risk_guard, self._default_risk_guard, step_dt, hierarchy, indicator_pool, sim_data, port_returns)

    def main(self, step_dt, hierarchy: dict, indicator_pool: dict, port_returns: dict, key) -> bool:
        # Called every datetime for each model and asset
        # Returns True if model can operate now
        return self._call(self._fn_main, self._default_main, step_dt, hierarchy, indicator_pool, port_returns, key)
    
#||=========================================================================================||

    



    # def __repr__(self):
    #     return f"<{self.__class__.__name__} name={self.name} capital={self.capital}>"