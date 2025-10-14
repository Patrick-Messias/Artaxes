import pandas as pd
from Indicator import Indicator

# macdv = MACDV.calculate(df, fast=12, slow=26, signal=9, vol_window=20, price_col='close')
class MACDV(Indicator): 
    @staticmethod
    def calculate(
        df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, vol_window: int = 20, price_col: str = 'close') -> pd.DataFrame:
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
    