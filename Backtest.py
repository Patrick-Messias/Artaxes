import BaseClass, uuid
from dataclasses import dataclass

@dataclass
class BacktestParams():
    name: str=f'backtest_{str(uuid.uuid4())}'

class Backtest(BaseClass)
    def __init__(self, backtest_params: BacktestParams):
        super().__init__()
        self.name = backtest_params.name

    def run():
        return None

    def check():
        return None

    def get_results():
        return None

    def get_statistics():
        return None




















