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
from typing import Literal, Dict, Optional, List
from dataclasses import dataclass, field
from Indicator import Indicator
from BaseClass import BaseClass, BaseManager

@dataclass
class MoneyManagerParams:
    name: str = field(default_factory=lambda: f'mm_{uuid.uuid4()}')

    capital: float=100000.0
    max_capital_exposure: float=1.0
    
    reb_frequency: Literal["tick", "daily", "weekly", "monthly", "yearly", "never"] = "weekly"
 
    # Dados externos para MM (Ex: volatilidade do mercado, regime de juros)
    # Agora usa Polars DataFrame
    assets: Dict[str, pl.DataFrame] = field(default_factory=dict)

    # Customizable parameters for specific System Managers (Ex: thresholds para desativar modelos, regras de ativação, etc)
    params: Dict = field(default_factory=dict) 
    
    # Indicadores específicos para balanceamento de ativos/modelos
    indicators: Optional[Dict[str, Indicator]] = field(default_factory=dict) 

class MoneyManager(BaseClass, BaseManager): # Classe base para SMM, MMM e PMM
    def __init__(self, mm_params: MoneyManagerParams):
        super().__init__()
        self.name = mm_params.name
        self.reb_frequency = mm_params.reb_frequency
        
        # Custom Rules & Data
        self.assets = mm_params.assets
        self.params = mm_params.params
        self.indicators = mm_params.indicators

    # ── SM Rebalance Func ───────────────────────────────────────────────────────

    def allocate(self, context: dict) -> Dict[str, float]:
        # Ranks each model by metric defined in model_hierarchy. Returns dict[model_name: score]
        return self._call(self._fn_allocate, self._default_allocate, context)

    def size(self, context: dict) -> List[str]:
        # Removes models that don't pass the filter function
        # Returns list of model_names that are active
        return self._call(self._fn_size, self._default_size, context)

    def risk_guard(self, context: dict) -> List[str]:
        # Orchestrates rank -> filter -> selection
        # Returns ordered list of active models
        return self._call(self._fn_risk_guard, self._default_risk_guard, context)

    #||=========================================================================================||

    



    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name} capital={self.capital}>"