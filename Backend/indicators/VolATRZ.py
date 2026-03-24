import pandas as pd
import numpy as np

class VolATRZ:
    """
    Calcula:
        - ATR
        - ATR percentual
        - Z-score do volume
        - Z-score do ATR percentual
        - Produto vol_atr_z

    Parâmetros
    ----------
    window : int
        Janela de cálculo do ATR e das médias móveis
    price_col : str
        Coluna de preço para normalização
    """

    def __init__(self, asset=None, timeframe=None, **params):
        super().__init__(asset, timeframe, **params)

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        window = self.params.get('window', 21)
        price_col = self.params.get('close', True)
        df = df.copy()

        if 'volume' not in df.columns:
            raise ValueError('Volume column required.')

        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df[price_col].shift()).abs()
        low_close = (df['low'] - df[price_col].shift()).abs()

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window).mean()

        # ATR%
        df['atr_pct'] = df['atr'] / df[price_col]

        # Z-score volume
        df['volume_z'] = (df['volume'] - df['volume'].rolling(window).mean()) / df['volume'].rolling(window).std()

        # Z-score ATR%
        df['atr_z'] = (df['atr_pct'] - df['atr_pct'].rolling(window).mean()) / df['atr_pct'].rolling(window).std()

        # Feature composta
        df['vol_atr_z'] = df['volume_z'] * df['atr_z']

        return df['vol_atr_z']
