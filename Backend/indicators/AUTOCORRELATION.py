import pandas as pd, numpy as np
from Indicator import Indicator

class ReturnAutocorrelation(Indicator):
    # Autocorrelation of returns over a rolling window.
    # - Positive autocorrelation: momentum regime
    # - Negative autocorrelation: mean reversion regime

    def __init__(self, asset=None, timeframe: str=None, **params):
        super().__init__(asset, timeframe, **params)

    def calculate(self, df: pd.DataFrame):
        window = self.params.get('window', 21)
        column_name = self.params.get('column_name', 'ret')

        # Retornos
        if column_name not in df.columns:
            if column_name == 'ret':
                df['ret'] = df['close'].pct_change().fillna(0)
            else: raise ValueError(f"DataFrame needs column '{column_name}'.")

        # Autocorrelação com lag=1 (r_t vs r_{t-1}), usa método rolling.corr para cálculo móvel
        autocorr = df['ret'].rolling(window).corr(df['ret'].shift(1))

        # Substitui NaN pelo neutro (0)
        autocorr = autocorr.fillna(0)

        return autocorr
