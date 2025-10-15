import pandas as pd
from Indicator import Indicator

class MA(Indicator):
    def __init__(self, asset, timeframe: str, window: int = 20, ma_type: str = 'sma', price_col: str = 'close'):
        super().__init__(asset, timeframe)
        self.window = window
        self.ma_type = ma_type 
        self.price_col = price_col

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        if self.ma_type == 'sma':
            return df[self.price_col].rolling(window=self.window, min_periods=self.window).mean()
        elif self.ma_type == 'ema':
            return df[self.price_col].ewm(span=self.window, adjust=False).mean()
        else:
            raise ValueError(f"Unsupported MA type: {self.ma_type}")




