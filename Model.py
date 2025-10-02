from typing import Dict, Optional, Callable
from dataclasses import dataclass, field
import BaseClass, Indicator

@dataclass
class Model_Parameters():
    name: str='unnamed_model'
    description: str=None

    assets: Union[Asset, Asset_Portfolio]=None # Asset(s) that the model will trade with it's strat(s)
    strat: dict=None

    indicators: Optional[Dict[str, Indicator]] = field(default_factory=dict) # For Strat selection management (if Strat > 1)
    model_rules: Optional[Dict[str, Callable]] = field(default_factory=dict)

class Model(BaseClass): 
    def __init__(self, model_params: Model_Parameters):
        super().__init__()
        self.name = model_params.name
        self.description = model_params.description
        
        self.strat = model_params.strat
        self.assets = model_params.assets

        # Custom Rules
        self.indicators = model_params.indicators
        self.model_rules = model_params.model_rules




    