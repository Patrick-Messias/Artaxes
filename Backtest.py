from dataclasses import dataclass, field
from BaseClass import BaseClass
from Trade import Trade
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
    

    def _calculate_ohlc_from_trade(self, trades: list[Trade]): # Gets trade's entry and end datetime's from Asset, then calculates open(strat), max(), min() and close(end) from that whole period

        return None




















