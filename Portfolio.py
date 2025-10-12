# Holds >1 models, doesn't define Assets, Server uniquely to Manage Positions between multiple models has to dominate over all MMM and MMA

from dataclasses import dataclass, field
from typing import Dict, Optional
from BaseClass import BaseClass
from PortfolioSystemManager import PortfolioSystemManager
from PortfolioMoneyManager import PortfolioMoneyManager
import uuid

@dataclass
class PortfolioParams():
    name: str = field(default_factory=lambda: f'model_{uuid.uuid4()}')
    models: dict=None
    portfolio_money_manager: Optional['PortfolioMoneyManager'] = None
    portfolio_system_manager: Optional['PortfolioSystemManager'] = None

class Portfolio(BaseClass): 
    def __init__(self, portfolio_params: PortfolioParams):
        super().__init__()
        self.name = portfolio_params.name
        self.models = portfolio_params.models
        self.portfolio_money_manager = portfolio_params.portfolio_money_manager
        self.portfolio_system_manager = portfolio_params.portfolio_system_manager

    def get_all_models(self) -> dict:
        return self.models if self.models else {}























