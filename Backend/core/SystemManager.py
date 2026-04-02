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
from typing import Literal, Dict, Optional, Callable
from dataclasses import dataclass, field
from Indicator import Indicator
from BaseClass import BaseClass

@dataclass
class SystemManagerParams:
    name: str = field(default_factory=lambda: f'sm_{uuid.uuid4()}')

    reb_frequency: Literal["tick", "daily", "weekly", "monthly", "yearly", "never"] = "weekly"
    
    # Dados externos para o System Manager (Ex: Calendário Econômico, Sentimento, CDT)
    # Migrado para usar dicionário de Polars DataFrames
    sm_assets: Dict[str, pl.DataFrame] = field(default_factory=dict)

    # Customizable parameters for specific System Managers (Ex: thresholds para desativar modelos, regras de ativação, etc)
    sm_params: Dict = field(default_factory=dict) 
    
    # Indicadores administrativos (Ex: Medidores de Regime de Mercado)
    sm_indicators: Optional[Dict[str, Indicator]] = field(default_factory=dict)
    

class SystemManager(BaseClass): 
    def __init__(self, system_params: SystemManagerParams):
        super().__init__()
        self.name = system_params.name
        self.reb_frequency = system_params.reb_frequency
        
        # Custom Data & Rules
        self.sm_assets = system_params.sm_assets
        self.sm_params = system_params.sm_params
        self.sm_indicators = system_params.sm_indicators

    def get_schedule(self, timeline: list) -> set:
        freq = self.reb_frequency 

        if not freq or freq == "never": 
            return pl.DataFrame({"ts": None}) # Updates every datetime

        df = pl.DataFrame({"ts": timeline})

        if freq == "tick":
            return df # Will always run

        if freq == "daily":
            condition = pl.col("ts").dt.date() != pl.col("ts").dt.date().shift(1)
        if freq == "weekly":
            condition = pl.col("ts").dt.week() != pl.col("ts").dt.week().shift(1)
        elif freq == "monthly":
            condition = pl.col("ts").dt.month() != pl.col("ts").dt.month().shift(1)
        elif freq == "yearly":
            condition = pl.col("ts").dt.year() != pl.col("ts").dt.year().shift(1)
        else:
            return set()

        # Fist candle is always a point of rebalance (start)
        return set(df.filter(condition | pl.col("ts").is_first())["ts"].to_list())

    # def should_execute(self, asset_name: str, strategy_name: str, context_df: Optional[pl.DataFrame] = None) -> bool:
    #     """
    #     Método central para decidir se uma operação deve prosseguir.
    #     Pode ser expandido nas subclasses para checar regras globais.
    #     """
    #     # Exemplo de lógica base: se não houver regras, libera geral (True)
    #     if not self.sm_rules:
    #         return True
        
    #     # Aqui as subclasses (ex: ModelSystemManager) implementariam a iteração sobre sm_rules
    #     return True

    # def filter_signals(self, signals_df: pl.DataFrame) -> pl.DataFrame:
    #     """
    #     Aplica filtros em massa sobre um DataFrame de sinais usando Polars.
    #     Útil para desativar sinais em horários de notícias ou regimes específicos.
    #     """
    #     # Exemplo: filter_signals poderia fazer um join_asof com dados externos (CDT)
    #     # e filtrar linhas onde a flag 'market_is_open' é falsa.
    #     return signals_df

    # =========================================================================================||

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name}>"
    





    # Management Indicators
    def fama_french(): # Imports all T-Bills, Assets, etc
        
        pass

























