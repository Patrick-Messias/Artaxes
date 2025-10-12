"""
# Money Management Algorithm (SMM / MMM / PMM) - Base class for all Money Management
Função: controlar risco, exposição e alocação de capital.
Camadas:
SMM (Strategy Money Management): define quanto alocar por trade dentro da estratégia (ex: 2% por sinal).
MMM (Model Money Management): define quanto cada estratégia do modelo recebe (ex: Strat A = 60%, Strat B = 40%).
PMM (Portfolio Money Management): define quanto cada modelo recebe do portfólio (ex: Model Momentum = 70%, Model MeanReversion = 30%).
🔹 Dominância: apenas o nível mais alto ativo (ex: PMM) sobrepõe os inferiores. Se PMM está ativo, ele comanda e os demais seguem as proporções internas.

IMPORTANTE: Strat, Model e Portfolio Manager são opcionais em cada nível, para economizar memória pode usar alguns ou nenhum, com um método basico para testar /
    criar na hora de verificar o StratMoneyManager na hora de realizar os backtests e usar um padrão fixo

"""

from typing import Dict, Optional, Callable
from dataclasses import dataclass, field
from BaseClass import BaseClass
from Indicator import Indicator
import uuid
import pandas as pd

@dataclass
class MoneyManagerParams:
    name: str = field(default_factory=lambda: f'mm_{uuid.uuid4()}')
    
    # Capital Management
    capital: float = 100000.0
    max_capital_exposure: float = 1.0
    
    # Drawdown Risk (Drawdown is oriented based on $ Money)
    drawdown: dict = field(default_factory=lambda: {"method": "var", "global": None, "monthly": None, "weekly": None, "daily": None}) # fixed
    
    # Indicators to find MM
    mm_external_data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    mm_indicators: Optional[Dict[str, Indicator]] = field(default_factory=dict) # For Model/Asset Balancing
    mm_rules: Optional[Dict[str, Callable]] = field(default_factory=dict)

class MoneyManager(BaseClass): # Base class for MMA, MMM and PMM
    def __init__(self, mm_params: MoneyManagerParams): # PMM(Portfolio) > MMM(Model) > MMA(Strat)
        self.name = mm_params.name
        
        # Capital Management
        self.capital = mm_params.capital
        self.max_capital_exposure = mm_params.max_capital_exposure

        # Drawdown Risk
        self.drawdown = mm_params.drawdown
        if self.drawdown["method"] not in ["var", "fixed"]:
            raise ValueError("Invalid drawdown method - Has to be 'var' or 'fixed'")
        if self.drawdown["method"] == "var":
            for period in ["global", "monthly", "weekly", "daily"]:
                if mm_params.drawdown[period] is not None and (mm_params.drawdown[period] <= 0 or mm_params.drawdown[period] >= 1):
                    raise ValueError(f"Invalid drawdown {period} - Has to be between 0 and 1 for 'var' method")
                
        # Custom Rules
        self.mm_external_data = mm_params.mm_external_data
        self.mm_indicators = mm_params.mm_indicators
        self.mm_rules = mm_params.mm_rules

    def calculate_var(self, confidence_level: float=0.5):
        return None



























