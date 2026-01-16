import pandas as pd, numpy as np
from Indicator import Indicator

class HURST(Indicator):
    def __init__(self, asset=None, timeframe=None, **params):
        super().__init__(asset, timeframe, **params)

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        price_col = self.params.get('price_col', 'pct_change')
        window = self.params.get('window', 63)
        calc_type = self.params.get('type', 'simple')

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

        if price_col == 'pct_change' and 'pct_change' not in df.columns:
            price_series = df['close'].pct_change().fillna(0)
        elif price_col == 'close':
            price_series = df['close']
        elif price_col == 'log_returns':
            price_series = np.log(df['close']).diff().fillna(0)
        else:
            raise ValueError(f"Coluna não suportada: {price_col}")

        if calc_type == 'simple':
            return price_series.rolling(window=window).apply(simple_hurst, raw=True)
        elif calc_type == 'rs':
            return price_series.rolling(window=window).apply(hurst_rs, raw=True)
        else:
            raise ValueError(f"Tipo não suportado: {calc_type}")







# class HURST(Indicator):
#     def __init__(self, asset=None, timeframe: str=None, window: int = 63, type: str = 'simple', price_col: str = 'pct_change'):
#         super().__init__(asset, timeframe, window = window, type = type.lower(), price_col = price_col)

#     def calculate(self, df: pd.DataFrame) -> pd.Series:
#         price_col = self.params.get('price_col', 'close')
#         window = self.params.get('window', 63)
#         type = self.params.get('type')

#         def simple_hurst(x):
#             n = len(x)
#             if n < 2:
#                 return np.nan
#             deviations = x - np.mean(x)
#             cumulative_dev = np.cumsum(deviations)
#             R = np.max(cumulative_dev) - np.min(cumulative_dev)
#             S = np.std(x, ddof=1)
#             if S == 0 or R == 0:
#                 return 0.5
#             return np.log(R/S) / np.log(n)

#         def hurst_rs(x, min_lag=2, max_lag=20):
#             n = len(x)
#             lags = range(min_lag, min(max_lag, n//2))
#             if len(lags) < 2:
#                 return np.nan
#             tau = [np.std(np.subtract(x[lag:], x[:-lag])) for lag in lags]
#             poly = np.polyfit(np.log(lags), np.log(tau), 1)
#             return poly[0] * 2.0

#         if price_col == 'pct_change' and 'pct_change' not in df.columns:
#             df['pct_change'] = df['close'].pct_change().fillna(0)

#         if type == 'simple':
#             return df[price_col].rolling(window=window).apply(simple_hurst, raw=True)
#         elif type == 'rs':
#             return df[price_col].rolling(window=window).apply(hurst_rs, raw=True)
#         else:
#             raise ValueError(f"Unsupported type: {type}")

