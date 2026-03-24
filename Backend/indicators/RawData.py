import polars as pl
from Indicator import Indicator

class RawData(Indicator):
    def __init__(self, asset: str, timeframe: str, price_col: str = 'close'):
        super().__init__()
        self.asset = asset
        self.timeframe = timeframe
        self.params = {'price_col': price_col}

    def calculate(self, df: pl.DataFrame, **kwargs) -> pl.Series:
        # Apenas retorna a coluna solicitada do dataframe de origem
        return df[self.params['price_col']]
