import pandas as pd
from Indicator import Indicator

class MACDV(Indicator): 
    def __init__(self, asset, timeframe: str, fast: int = 12, slow: int = 26, signal: int = 9, vol_window: int = 20, price_col: str = 'close'):
        super().__init__(asset, timeframe, fast=fast, slow = slow, signal = signal, vol_window = vol_window, price_col = price_col)

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:  
        fast = self.params.get('fast')
        slow = self.params.get('slow')
        signal = self.params.get('signal')
        vol_window = self.params.get('vol_window')
        price_col = self.params.get('price_col')

        # MACD
        ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
        ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal

        # Vol
        returns = df[price_col].pct_change()
        volatility = returns.rolling(window=vol_window).std()

        # MACDV: MACD normalized by vol
        macdv = macd / volatility

        return pd.DataFrame({
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_hist': macd_hist,
            'macdv': macdv
        })
    