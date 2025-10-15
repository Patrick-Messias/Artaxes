from dataclasses import dataclass, field
from BaseClass import BaseClass
import uuid

@dataclass
class WalkforwardParams():
    name: str = field(default_factory=lambda: f'model_{uuid.uuid4()}')
    method: str = 'simple' # 'simple' checks trade that where opened and close at a date | 'complete' runs full backtest and only trades at selected window
    
class Walkforward(BaseClass):
    def __init__(self, wf_params: WalkforwardParams = WalkforwardParams()):
        super().__init__()
        self.name = wf_params.name
        self.method = wf_params.method

    def run(self):
        return None
























