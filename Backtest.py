from dataclasses import dataclass, field
from BaseClass import BaseClass
import uuid

@dataclass
class BacktestParams():
    name: str = field(default_factory=lambda: f'model_{uuid.uuid4()}')

class Backtest(BaseClass):
    def __init__(self, backtest_params: BacktestParams = BacktestParams()):
        super().__init__()
        self.name = backtest_params.name

    def run(self):
        return None

    def check(self):
        return None

    def get_results(self):
        return None

    def get_statistics(self):
        return None




















