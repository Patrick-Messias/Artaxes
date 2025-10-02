from typing import Dict, Optional, Union, List, Callable
from dataclasses import dataclass, field
from BMM import Base_Management_Algorithm
import Indicator

@dataclass
class Model_Money_Management_Params:
    """Parâmetros para configurar o Money Management"""
    name: str='unnamed_mmm'

class Money_Management_Algorithm(Base_Management_Algorithm): # Manages Model's risk and money management
    def __init__(self, mmm_params: Model_Money_Management_Params): # PMM(Portfolio) > MMM(Model) > MMA(Strat)
        super().__init__()
        self.name: mmm_params.name

    def calculate_model_capital_allocation(self, capital: float, strats: dict): # Rebalances capital allocation between Strats enside the Model with PMM's capital allocated to Model
        return strats