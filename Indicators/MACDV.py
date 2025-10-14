import pandas as pd
from Indicator import Indicator

# macdv = MACDV.calculate(df, fast=12, slow=26, signal=9, vol_window=20, price_col='close')
class MACDV(Indicator): 
    def __init__(self, timeframe: str, fast: int = 12, slow: int = 26, signal: int = 9, vol_window: int = 20, price_col: str = 'close'):
        super().__init__(timeframe)
        self.fast = fast
        self.slow = slow    
        self.signal = signal
        self.vol_window = vol_window
        self.price_col = price_col

    @staticmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:        
        # MACD
        ema_fast = df[self.price_col].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df[self.price_col].ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=self.signal, adjust=False).mean()
        macd_hist = macd - macd_signal

        # Vol
        returns = df[self.price_col].pct_change()
        volatility = returns.rolling(window=self.vol_window).std()

        # MACDV: MACD normalized by vol
        macdv = macd / volatility

        return pd.DataFrame({
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_hist': macd_hist,
            'macdv': macdv
        })
    