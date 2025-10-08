from typing import Dict, Optional, Callable
from dataclasses import dataclass, field
import BaseClass, Indicator, uuid
from System_Management_Algorithm import System_Management_Algorithm

@dataclass
class PMA_Parameters():
    name: str=f'pma_{str(uuid.uuid4())}'
    
    model_hierarchy: str='default'
    rebalance_frequency: str='weekly'

    external_data: Dict[str, pd.DataFrame] = field(default_factory=dict)=None

    indicators: Optional[Dict[str, Indicator]] = field(default_factory=dict) # For Model/Asset Balancing
    pma_rules: Optional[Dict[str, Callable]] = field(default_factory=dict)

class Portfolio_Manager_Algorithm(System_Management_Algorithm): # Manages portfolio's model hierarchy 
    def __init__(self, pma_params: PMA_Parameters):
        super().__init__()
        self.name = pma_params.name
        
        self.model_hierarchy = pma_params.model_hierarchy
        self.rebalance_frequency = pma_params.rebalance_frequency

        self.external_data = pma_params.external_data # For external data like CDT

        # Custom Rules
        self.indicators = pma_params.indicators
        self.pma_rules = pma_params.pma_rules

    def update_model_hierarchy(self): # Updates the hierarchy of the models enside the portfolio

        # Checks pma_rules on how to rebalance

        # Analyses current hierarchy and models performance based on rebalancing rules

        # Updates hierachy

        return 






















