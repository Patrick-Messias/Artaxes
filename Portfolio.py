# Holds >1 models, doesn't define Assets, Server uniquely to Manage Positions between multiple models has to dominate over all MMM and MMA

from dataclasses import dataclass
import BaseClass, uuid
from PortfolioManager import PortfolioManager
from PortfolioMoneyManager import PortfolioMoneyManager

@dataclass
class Portfolio_Parameters():
    name: str = field(default_factory=lambda: f'model_{uuid.uuid4()}')
    models: dict=None
    portfolio_money_manager: Optional['PortfolioMoneyManager'] = None
    portfolio_system_manager: Optional['PortfolioManager'] = None

class Portfolio(BaseClass): 
    def __init__(self, portfolio_params: Portfolio_Parameters):
        super().__init__()
        self.name = portfolio_params.name
        self.models = portfolio_params.models
        self.portfolio_money_manager = portfolio_params.portfolio_money_manager
        self.portfolio_system_manager = portfolio_params.portfolio_system_manager

    def get_all_models(self) -> dict:
        return self.models if self.models else {}























