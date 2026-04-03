"""
# System Management Algorithm (SMA) - Base class for all System Management
Função: orquestrar o comportamento do sistema em tempo de execução.
Liga/desliga Strats ou Models conforme regras globais.
Define quais combinações (Model + Asset + Strat) estão ativas.
Pode implementar lógica de auto-adaptação (ex: desativar modelos com drawdown alto).
Atua sobre os níveis superiores (controla quem “fala” com o PMM e o TM).
"""

import polars as pl, uuid
from typing import Literal, Dict, Optional
from dataclasses import dataclass, field
from Indicator import Indicator
from BaseClass import BaseClass, BaseManager

@dataclass
class SystemManagerParams:
    name: str = field(default_factory=lambda: f'sm_{uuid.uuid4()}')

    reb_frequency: Literal["tick", "daily", "weekly", "monthly", "yearly", "never"] = "weekly"
    
    # Dados externos para o System Manager (Ex: Calendário Econômico, Sentimento, CDT)
    # Migrado para usar dicionário de Polars DataFrames
    assets: Dict[str, pl.DataFrame] = field(default_factory=dict)

    # Customizable parameters for specific System Managers (Ex: thresholds para desativar modelos, regras de ativação, etc)
    params: Dict = field(default_factory=dict) 
    
    # Indicadores administrativos (Ex: Medidores de Regime de Mercado)
    indicators: Optional[Dict[str, Indicator]] = field(default_factory=dict)
    

class SystemManager(BaseClass, BaseManager): 
    def __init__(self, system_params: SystemManagerParams):
        super().__init__()
        self.name = system_params.name
        self.reb_frequency = system_params.reb_frequency
        
        # Custom Data & Rules
        self.assets = system_params.assets
        self.params = system_params.params
        self.indicators = system_params.indicators

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name}>"








    #||=========================================================================================||

    # Management Indicators
    def fama_french(): pass # Imports all T-Bills, Assets, etc
        
    def mae_mpe(): pass # 
    























