import pandas as pd
import numpy as np
from Indicator import Indicator

class HARComponents(Indicator):
    # HAR (Heterogeneous AutoRegressive) components.
    # Decompõe a volatilidade realizada (RV) em componentes diário, semanal e mensal,
    # usados para capturar a estrutura temporal da volatilidade (Corsi, 2009).

    def __init__(self, asset=None, timeframe: str=None, price_col: str='close'):
        super().__init__(asset, timeframe)
        self.price_col = price_col

    def calculate(self, df: pd.DataFrame):
        if self.price_col not in df.columns:
            raise ValueError(f"Coluna '{self.price_col}' não encontrada no DataFrame.")

        # 1️⃣ Log-retornos
        log_ret = np.log(df[self.price_col] / df[self.price_col].shift(1))
        rv = (log_ret ** 2).fillna(0)  # Realized volatility

        # 2️⃣ Componentes HAR
        har_df = pd.DataFrame({
            'har_daily': rv,
            'har_weekly': rv.rolling(5).mean(),
            'har_monthly': rv.rolling(22).mean()
        }, index=df.index)

        # 3️⃣ Normalização (z-score) apenas dos componentes HAR
        har_df = (har_df - har_df.mean()) / (har_df.std() + 1e-8)
        return har_df