import pandas as pd, numpy as np
from Indicator import Indicator

class RSIZScore(Indicator):
    def __init__(self, timeframe: str, rsi_window: int = 14, zscore_window: int = 14, price_col: str = 'close'):
        super().__init__(timeframe)
        self.rsi_window = rsi_window
        self.zscore_window = zscore_window
        self.price_col = price_col

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        delta = df[self.price_col].diff().fillna(0)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=self.rsi_window, min_periods=self.rsi_window).mean()
        avg_loss = pd.Series(loss).rolling(window=self.rsi_window, min_periods=self.rsi_window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Calcula o ZScore do RSI
        rsi_zscore = (rsi - rsi.rolling(window=self.zscore_window).mean()) / rsi.rolling(window=self.zscore_window).std()

        return rsi_zscore
    