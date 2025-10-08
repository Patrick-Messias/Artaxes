from typing import Dict, Optional, Callable
from dataclasses import dataclass, field
from BMM import Base_Management_Algorithm
import Indicator, uuid

@dataclass
class Portfolio_Money_Management_Params:
    """Parâmetros para configurar o Money Management"""
    name: str=f'pmm_{str(uuid.uuid4())}'
    capital_allocation: np.array=None

class Money_Management_Algorithm(Base_Management_Algorithm): # Manages Portfolio's risk and money management
    def __init__(self, pmm_params: Portfolio_Money_Management_Params): 
        super().__init__()
        self.name: pmm_params.name
        self.capital_allocation: pmm_params.capital_allocation




















