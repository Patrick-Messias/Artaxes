from dataclasses import dataclass, field
from BaseClass import BaseClass
import uuid

@dataclass
class WalkforwardParams():
    name: str = field(default_factory=lambda: f'model_{uuid.uuid4()}')
    method: str = 'simple' # 'simple' checks trade that where opened and close at a date | 'complete' runs full backtest and only trades at selected window
    isos: list[str] | None = None # List of in-sample/out-of-sample periods to run, e.g., ['12_12', '6_6']
    
class Walkforward(BaseClass):
    def __init__(self, wf_params: WalkforwardParams = WalkforwardParams()):
        super().__init__()
        self.name = wf_params.name
        self.method = wf_params.method
        self.isos = wf_params.isos

    def run(self):
        return None






    # NOTE Apenas gerar todos os trades, salvnado informações necessárias para a operação, depois analisar wf/portfolio,etc

















