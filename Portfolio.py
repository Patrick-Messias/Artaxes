from dataclasses import dataclass
import BaseClass

@dataclass
class Portfolio_Parameters():
    name: str='unnamed_portfolio'
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























