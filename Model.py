 # Holds 1+ Strat with 1+ Asset, uses Model Management Algorith to select from entries from multiple Strat or Asset positions (if either >1) 
 # and Model Money Management to manage Risk, Exposition, etc.
 
from typing import Dict, Optional, Callable, Union
from dataclasses import dataclass, field
from Asset import Asset, AssetParams, Asset_Portfolio
from Indicator import Indicator
from BaseClass import BaseClass
from ModelMoneyManager import ModelMoneyManager, ModelMoneyManagerParams
from ModelSystemManager import ModelSystemManager, ModelSystemManagerParams
from MoneyManager import MoneyManagerParams
import uuid

@dataclass
class Model_Parameters():
    name: str = field(default_factory=lambda: f'model_{uuid.uuid4()}')
    description: str=None

    assets: Union[Asset, Asset_Portfolio]=None # Asset(s) that the model will trade with it's strat(s)
    strat: dict=None

    execution_timeframe=None
    model_money_manager: Optional[ModelMoneyManager] = None
    model_system_manager: Optional[ModelSystemManager] = None

class Model(BaseClass):
    def __init__(self, model_params: Model_Parameters):
        super().__init__()
        self.name = model_params.name
        self.description = model_params.description
        
        self.strat = model_params.strat
        self.assets = model_params.assets
        
        self.execution_timeframe = model_params.execution_timeframe
        
        # ModelMoneyManager is optional - if None, will use default or PMM from Operation
        self.model_money_manager = model_params.model_money_manager
        
        # ModelSystemManager is optional - if None, will use default system management
        self.model_system_manager = model_params.model_system_manager
        





    