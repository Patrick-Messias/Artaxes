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
from typing import Dict, Optional, Callable, Union
from dataclasses import dataclass, field
from BaseClass import BaseClass
from Indicator import Indicator

@dataclass
class MoneyManagerParams:
    name: str = field(default_factory=lambda: f'mm_{uuid.uuid4()}')
    
    # Capital Management
    capital: float = 100000.0
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
    mm_external_data: Dict[str, pl.DataFrame] = field(default_factory=dict)
    
    # Indicadores específicos para balanceamento de ativos/modelos
    mm_indicators: Optional[Dict[str, Indicator]] = field(default_factory=dict) 
    
    # Regras customizadas de alocação
    mm_rules: Optional[Dict[str, Callable]] = field(default_factory=dict)

class MoneyManager(BaseClass): # Classe base para SMM, MMM e PMM
    def __init__(self, mm_params: MoneyManagerParams):
        super().__init__()
        self.name = mm_params.name
        
        # Capital Management
        self.capital = mm_params.capital
        self.max_capital_exposure = mm_params.max_capital_exposure

        # Drawdown Risk Validation
        self.drawdown = mm_params.drawdown
        self._validate_drawdown_settings()
                
        # Custom Rules & Data
        self.mm_external_data = mm_params.mm_external_data
        self.mm_indicators = mm_params.mm_indicators
        self.mm_rules = mm_params.mm_rules

    def _validate_drawdown_settings(self):
        """Valida se os limites de drawdown estão coerentes com o método escolhido."""
        if self.drawdown["method"] not in ["var", "fixed"]:
            raise ValueError("Invalid drawdown method - Has to be 'var' or 'fixed'")
            
        if self.drawdown["method"] == "var":
            for period in ["global", "monthly", "weekly", "daily"]:
                val = self.drawdown.get(period)
                if val is not None and (val <= 0 or val >= 1):
                    raise ValueError(f"Invalid drawdown {period} - Value {val} must be between 0 and 1 for 'var' (percentage) method")

    def calculate_var(self, confidence_level: float = 0.95):
        """
        Placeholder para cálculo de Value at Risk.
        No Polars, isso seria feito operando sobre os retornos históricos 
        contidos no mm_external_data.
        """
        # Exemplo teórico de uso com Polars:
        # returns = self.mm_external_data['equity_curve'].select(pl.col('returns'))
        # return returns.quantile(1 - confidence_level)
        return None

    def get_allocated_capital(self) -> float:
        """Retorna o capital máximo que este manager pode expor."""
        return self.capital * self.max_capital_exposure

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name} capital={self.capital}>"