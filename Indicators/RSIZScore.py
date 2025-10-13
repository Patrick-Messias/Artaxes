import pandas as pd, numpy as np

# rsiz = RSIZscore.calculate(df, rsi_window=14, zscore_window=20, price_col='close')
class RSIZscore:
    @staticmethod 
    def calculate(df: pd.DataFrame, rsi_window: int = 14, zscore_window: int = 14, price_col: str = 'close') -> pd.Series:
        delta = df[price_col].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=rsi_window, min_periods=rsi_window).mean()
        avg_loss = pd.Series(loss).rolling(window=rsi_window, min_periods=rsi_window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Calcula o ZScore do RSI
        rsi_zscore = (rsi - rsi.rolling(window=zscore_window).mean()) / rsi.rolling(window=zscore_window).std()

        return rsi_zscore
    