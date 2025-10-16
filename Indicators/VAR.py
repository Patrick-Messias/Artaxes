import pandas as pd, numpy as np, scipy.stats as st
from Indicator import Indicator

class VAR(Indicator):
    def __init__(self, asset, timeframe: str, window: int = 20, alpha: float = 0.05, type: str = 'parametric', price_col: str = 'pct_change'):
        super().__init__(asset, timeframe)
        self.asset = asset
        self.window = window
        self.alpha = alpha 
        self.type = type.lower()
        self.price_col = price_col

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        if self.price_col == 'pct_change' and 'pct_change' not in df.columns:
            df['pct_change'] = df['close'].pct_change().fillna(0)

        if self.price_col not in df.columns:
            raise KeyError(f"Column '{self.price_col}' not found in DataFrame.")

        if len(df) < self.window:
            raise ValueError("DataFrame too short for selected window.")

        if self.type == 'parametric':
            rolling_mean = df[self.price_col].rolling(self.window).mean()
            rolling_std = df[self.price_col].rolling(self.window).std(ddof=1)
            z_score = st.norm.ppf(1 - self.alpha)
            var_series = rolling_mean - z_score * rolling_std
            return var_series

        else:
            raise ValueError(f"Unsupported type: {self.type}")

