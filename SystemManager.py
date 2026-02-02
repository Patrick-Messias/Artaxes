"""
# System Management Algorithm (SMA) - Base class for all System Management
Função: orquestrar o comportamento do sistema em tempo de execução.
Liga/desliga Strats ou Models conforme regras globais.
Define quais combinações (Model + Asset + Strat) estão ativas.
Pode implementar lógica de auto-adaptação (ex: desativar modelos com drawdown alto).
Atua sobre os níveis superiores (controla quem “fala” com o PMM e o TM).
"""

import polars as pl
import uuid
from typing import Dict, Optional, Callable
from dataclasses import dataclass, field
from BaseClass import BaseClass
from Indicator import Indicator

@dataclass
class SystemManagerParams:
    name: str = field(default_factory=lambda: f'sm_{uuid.uuid4()}')
    
    # Dados externos para o System Manager (Ex: Calendário Econômico, Sentimento, CDT)
    # Migrado para usar dicionário de Polars DataFrames
    sm_external_data: Dict[str, pl.DataFrame] = field(default_factory=dict)
    
    # Indicadores administrativos (Ex: Medidores de Regime de Mercado)
    sm_indicators: Optional[Dict[str, Indicator]] = field(default_factory=dict)
    
    # Regras lógicas de ativação/desativação (Filtros de sistema)
    sm_rules: Optional[Dict[str, Callable]] = field(default_factory=dict)

class SystemManager(BaseClass): 
    def __init__(self, system_params: SystemManagerParams):
        super().__init__()
        self.name = system_params.name
        
        # Custom Data & Rules
        self.sm_external_data = system_params.sm_external_data
        self.sm_indicators = system_params.sm_indicators
        self.sm_rules = system_params.sm_rules

    def should_execute(self, asset_name: str, strategy_name: str, context_df: Optional[pl.DataFrame] = None) -> bool:
        """
        Método central para decidir se uma operação deve prosseguir.
        Pode ser expandido nas subclasses para checar regras globais.
        """
        # Exemplo de lógica base: se não houver regras, libera geral (True)
        if not self.sm_rules:
            return True
        
        # Aqui as subclasses (ex: ModelSystemManager) implementariam a iteração sobre sm_rules
        return True

    def filter_signals(self, signals_df: pl.DataFrame) -> pl.DataFrame:
        """
        Aplica filtros em massa sobre um DataFrame de sinais usando Polars.
        Útil para desativar sinais em horários de notícias ou regimes específicos.
        """
        # Exemplo: filter_signals poderia fazer um join_asof com dados externos (CDT)
        # e filtrar linhas onde a flag 'market_is_open' é falsa.
        return signals_df

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name}>"