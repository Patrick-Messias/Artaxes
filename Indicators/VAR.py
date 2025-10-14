import pandas as pd, numpy as np, scipy.stats as st
from Indicator import IndicatorBase

class VAR(IndicatorBase):
    def __init__(self, asset, timeframe: str, window: int = 20, alpha: float = 0.05, type: str = 'parametric', price_col: str = 'pct_change'):
        super().__init__(timeframe)
        self.asset = asset
        self.window = window
        self.alpha = alpha 
        self.type = type.lower()
        self.price_col = price_col

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        def parametric_var(x):
            mean = np.mean(x)
            std_dev = np.std(x, ddof=1)
            z_score = st.norm.ppf(1 - self.alpha)
            return mean + z_score * std_dev

        if self.price_col == 'pct_change' and 'pct_change' not in df.columns:
            df['pct_change'] = df['close'].pct_change().fillna(0)

        if self.type == 'parametric':
            return df[self.price_col].rolling(window=self.window).apply(parametric_var, raw=True)
        else:
            raise ValueError(f"Unsupported type: {self.type}")

