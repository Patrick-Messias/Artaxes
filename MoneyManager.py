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

from typing import Dict, Optional, Union, List, Callable
from dataclasses import dataclass, field
import BaseClass, Indicator, uuid

@dataclass
class MoneyManagerParams:
    """Parâmetros para configurar o Money Management"""
    name: str = field(default_factory=lambda: f'sm_{uuid.uuid4()}')
    
    # Capital Management
    init_capital: float = 100000.0
    max_capital_exposure: float = 1.0
    leverage: float = 1.0
    compounding = None #'regular'
    
    # Trade Risk - Trade Management
    position_sizing_type: str = 'percentage'  # 'percentage', 'kelly', 'confidence'
    position_sizing_from: str = 'balance'     # 'balance', 'equity'
    position_sizing_method: str = 'regular'   # 'regular', 'dynamic'
    trade_risk_default: float = 0.01
    trade_risk_min: float = 0.001
    trade_risk_max: float = 0.05
    trade_max_num_open: int = 1
    trade_min_num_analysis: int = 100

    # Drawdown Risk
    drawdown_method = 'var' #'fixed'
    drawdown_global: float = None
    drawdown_monthly: float = None
    drawdown_weekly: float = None
    drawdown_daily: float = None
    
    # Advanced Parameters
    confidence_level: float = 0.5
    kelly_weight: float = 0.1
    
    # Indicators to find MMA
    indicators: Optional[Dict[str, Indicator]] = field(default_factory=dict) # For Model/Asset Balancing
    mma_rules: Optional[Dict[str, Callable]] = field(default_factory=dict)

class MoneyManager(BaseClass): # Base class for MMA, MMM and PMM
    def __init__(self, mma_params: MoneyManagerParams): # PMM(Portfolio) > MMM(Model) > MMA(Strat)
        self.name = mma_params.name
        
        # Capital Management
        self.init_capital = mma_params.init_capital
        self.max_capital_exposure = mma_params.max_capital_exposure
        self.leverage = mma_params.leverage

        # Trade Risk - Trade Management
        self.position_sizing_type = mma_params.position_sizing_type
        self.position_sizing_from = mma_params.position_sizing_from
        self.position_sizing_method = mma_params.position_sizing_method
        self.trade_risk_default = mma_params.trade_risk_default
        self.trade_risk_min = mma_params.trade_risk_min
        self.trade_risk_max = mma_params.trade_risk_max
        self.trade_max_num_open = mma_params.trade_max_num_open
        self.trade_min_num_analysis = mma_params.trade_min_num_analysis

        # Drawdown Risk
        self.drawdown_method = mma_params.drawdown_method
        self.drawdown_global = mma_params.drawdown_global
        self.drawdown_monthly = mma_params.drawdown_monthly
        self.drawdown_weekly = mma_params.drawdown_weekly
        self.drawdown_daily = mma_params.drawdown_daily
        
        # Advanced Parameters
        self.confidence_level = mma_params.confidence_level
        self.kelly_weight = mma_params.kelly_weight

        # Custom Rules
        self.indicators = mma_params.indicators
        self.mma_rules = mma_params.mma_rules

    def calculate_kelly_criterion(self, trades: dict[Trade], weight: int=0.1):
        return None

    def calculate_var(self):
        return None



























