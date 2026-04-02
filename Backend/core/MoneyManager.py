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
from typing import Literal, Dict, Optional, Callable
from dataclasses import dataclass, field
from Indicator import Indicator
from BaseClass import BaseClass

@dataclass
class MoneyManagerParams:
    name: str = field(default_factory=lambda: f'mm_{uuid.uuid4()}')
    
    reb_frequency: Literal["tick", "daily", "weekly", "monthly", "yearly", "never"] = "weekly"

    # Capital Management
    capital: float = 0.0
    max_capital_exposure: float = 1.0 # Ex: 1.0 = 100% do capital
    
    # Drawdown Risk (Orientado a valor financeiro ou percentual conforme o método)
    # Ex: {"method": "var", "global": 0.2} -> Risco de 20% do capital total
    drawdown: dict = field(default_factory=lambda: {
        "method": "var", 
        "global": None, 
        "monthly": None, 
        "weekly": None, 
        "daily": None
    }) 
    
    # Dados externos para MM (Ex: volatilidade do mercado, regime de juros)
    # Agora usa Polars DataFrame
    mm_assets: Dict[str, pl.DataFrame] = field(default_factory=dict)

    # Customizable parameters for specific System Managers (Ex: thresholds para desativar modelos, regras de ativação, etc)
    mm_params: Dict = field(default_factory=dict) 
    
    # Indicadores específicos para balanceamento de ativos/modelos
    mm_indicators: Optional[Dict[str, Indicator]] = field(default_factory=dict) 

class MoneyManager(BaseClass): # Classe base para SMM, MMM e PMM
    def __init__(self, mm_params: MoneyManagerParams):
        super().__init__()
        self.name = mm_params.name
        self.reb_frequency = mm_params.reb_frequency
        
        # Capital Management
        self.capital = mm_params.capital
        self.max_capital_exposure = mm_params.max_capital_exposure

        # Drawdown Risk Validation
        self.drawdown = mm_params.drawdown
        self._validate_drawdown_settings()
                
        # Custom Rules & Data
        self.mm_assets = mm_params.mm_assets
        self.mm_params = mm_params.mm_params
        self.mm_indicators = mm_params.mm_indicators

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







    def _validate_drawdown_settings(self):
        """Valida se os limites de drawdown estão coerentes com o método escolhido."""
        if self.drawdown["method"] not in ["var", "fixed"]:
            raise ValueError("Invalid drawdown method - Has to be 'var' or 'fixed'")
            
        if self.drawdown["method"] == "var":
            for period in ["global", "monthly", "weekly", "daily"]:
                val = self.drawdown.get(period)
                if val is not None and (val <= 0 or val >= 1):
                    raise ValueError(f"Invalid drawdown {period} - Value {val} must be between 0 and 1 for 'var' (percentage) method")

    def calculate_var(): pass # UTILIZAR CLASSE DO INDICADOR VAR PARA EVITAR REDUNDANCIA

    def get_allocated_capital(self) -> float:
        """Retorna o capital máximo que este manager pode expor."""
        return self.capital * self.max_capital_exposure

    # =========================================================================================||

    



    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name} capital={self.capital}>"