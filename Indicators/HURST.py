import pandas as pd, numpy as np
from Indicator import IndicatorBase

import pandas as pd, numpy as np
from Indicator import IndicatorBase

class HURST(IndicatorBase):
    def __init__(self, asset, timeframe: str, window: int = 20, type: str = 'simple', price_col: str = 'pct_change'):
        super().__init__(asset, timeframe)
        self.window = window
        self.type = type.lower()
        self.price_col = price_col

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        def simple_hurst(x):
            n = len(x)
            if n < 2:
                return np.nan
            deviations = x - np.mean(x)
            cumulative_dev = np.cumsum(deviations)
            R = np.max(cumulative_dev) - np.min(cumulative_dev)
            S = np.std(x, ddof=1)
            if S == 0 or R == 0:
                return 0.5
            return np.log(R/S) / np.log(n)

        def hurst_rs(x, min_lag=2, max_lag=20):
            n = len(x)
            lags = range(min_lag, min(max_lag, n//2))
            if len(lags) < 2:
                return np.nan
            tau = [np.std(np.subtract(x[lag:], x[:-lag])) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0

        if self.price_col == 'pct_change' and 'pct_change' not in df.columns:
            df['pct_change'] = df['close'].pct_change().fillna(0)

        if self.type == 'simple':
            return df[self.price_col].rolling(window=self.window).apply(simple_hurst, raw=True)
        elif self.type == 'rs':
            return df[self.price_col].rolling(window=self.window).apply(hurst_rs, raw=True)
        else:
            raise ValueError(f"Unsupported type: {self.type}")

