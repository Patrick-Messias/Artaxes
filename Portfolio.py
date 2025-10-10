# Holds >1 models, doesn't define Assets, Server uniquely to Manage Positions between multiple models has to dominate over all MMM and MMA

from dataclasses import dataclass
import BaseClass, uuid

@dataclass
class Portfolio_Parameters():
    name: str = field(default_factory=lambda: f'model_{uuid.uuid4()}')
    models: dict=None
    pma: Portfolio_Manager_Algorithm=None
    pmm: Portfolio_Money_Management=None

class Portfolio(BaseClass): 
    def __init__(self, portfolio_params: Portfolio_Parameters):
        super().__init__()
        self.name = portfolio_params.name
        self.models = portfolio_params.models
        self.pma = portfolio_params.pma
        self.pmm = portfolio_params.pmm

    def get_all_models(self) -> dict:
        return self.models if self.models else {}























