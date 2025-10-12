import uuid
from typing import Dict, Optional, Callable
from dataclasses import dataclass, field
from BaseClass import BaseClass
from Indicator import Indicator
from SystemManager import SystemManager, SystemManagerParams

@dataclass
class PortfolioSystemManagerParams(SystemManagerParams):
    model_hierarchy: dict = field(default_factory=lambda: {"order_by": 'highest', "metric": 'profit_perc'})
    rebalance_frequency: str = 'weekly'
    close_open_trades_on_rebalance: bool = False

class PortfolioSystemManager(SystemManager): # Manages portfolio's model hierarchy 
    def __init__(self, pm_params: PortfolioSystemManagerParams):
        super().__init__(pm_params) # SystemManager attributes init
        
        self.model_hierarchy = dict(pm_params.model_hierarchy)
        self.rebalance_frequency = pm_params.rebalance_frequency
        self.close_open_trades_on_rebalance = pm_params.close_open_trades_on_rebalance


    def update_model_hierarchy(self): # Updates the hierarchy of the models enside the portfolio

        # Checks pma_rules on how to rebalance

        # Analyses current hierarchy and models performance based on rebalancing rules

        # Updates hierachy

        return 






















