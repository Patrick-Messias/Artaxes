import pandas as pd, numpy as np
from Indicator import Indicator

class VIXFIX(Indicator):
    def __init__(self, asset, timeframe: str, window: int = 22, price_high_col: str = 'close', price_low_col: str = 'low'):
        super().__init__(asset, timeframe)
        self.window = window
        self.price_high_col = price_high_col
        self.price_low_col = price_low_col

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        # VixFix = (Highest(Close, 22)-Low) / (Highest(Close, 22)) * 100

        close = df[self.price_high_col]
        highest_close = close.rolling(window=self.window, min_periods=self.window).max()
        low = df[self.price_low_col]

        numerator = highest_close - low
        denominator = highest_close
        ratio = numerator / denominator

        return ratio * 100
    