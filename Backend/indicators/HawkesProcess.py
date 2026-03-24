import pandas as pd, numpy as np
from Indicator import Indicator

# Tipo de regime	Feature ideal (inclui Hawkes?)
# Volatilidade	Hawkes(np.log(high/low)), ou amplitude de candle
# Tendência	Hawkes(np.log(close).diff()), captando momentum autoexcitatório
# Timing / reversão	Hawkes(pct_change) com decaimento rápido (kappa ↑)

class HawkesProcess(Indicator):
    def __init__(self, asset=None, timeframe: str=None, kappa: float=0.01, price_col: str='close'):
        super().__init__(asset, timeframe)
        self.kappa = kappa
        self.price_col = price_col

    def calculate(self, df: pd.DataFrame):
        assert self.kappa > 0.0

        # Garante que há uma coluna de variação percentual se for a escolhida
        if 'pct_change' not in df.columns: df['pct_change'] = df[self.price_col].pct_change().fillna(0)

        alpha = np.exp(-self.kappa)
        arr = df[self.price_col].to_numpy()
        output = np.full(len(df), np.nan)

        for i in range(1, len(df)):
            if np.isnan(output[i-1]):
                output[i] = arr[i]
            else:
                output[i] = output[i-1] * alpha + arr[i]

        # Retorna uma série alinhada ao índice original
        return pd.Series(output * self.kappa, index=df.index)
