 # Holds 1+ Strat with 1+ Asset, uses Model Management Algorith to select from entries from multiple Strat or Asset positions (if either >1) 
 # and Model Money Management to manage Risk, Exposition, etc.
 
from typing import Dict, Optional, Callable
from dataclasses import dataclass, field
import BaseClass, Indicator

@dataclass
class Model_Parameters():
    name: str='unnamed_model'
    description: str=None

    assets: Union[Asset, Asset_Portfolio]=None # Asset(s) that the model will trade with it's strat(s)
    strat: dict=None

    execution_timeframe=None

    indicators: Optional[Dict[str, Indicator]] = field(default_factory=dict) # For Strat selection management (if Strat > 1)
    model_rules: Optional[Dict[str, Callable]] = field(default_factory=dict)

class Model(BaseClass):
    def __init__(self, model_params: Model_Parameters):
        super().__init__()
        self.name = model_params.name
        self.description = model_params.description
        
        self.strat = model_params.strat
        self.assets = model_params.assets
        
        self.execution_timeframe = model_params.execution_timeframe

        # Custom Rules
        self.indicators = model_params.indicators
        self.model_rules = model_params.model_rules




    