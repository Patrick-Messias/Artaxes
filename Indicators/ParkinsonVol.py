import pandas as pd, numpy as np
from Indicator import Indicator

class ParkinsonVolatility(Indicator):
    """
    Calcula a volatilidade de Parkinson:
    σ_Parkinson = sqrt( (1 / (4 * ln(2))) * mean( ln(high / low)^2 ) )

    - Usa apenas os preços high/low de cada candle.
    - É mais eficiente para medir volatilidade diária/intradiária.
    """

    def __init__(self, asset=None, timeframe: str=None, window: int=21):
        super().__init__(asset, timeframe)
        self.window = window

    def calculate(self, df: pd.DataFrame):
        # Evita divisões por zero ou valores inválidos
        hl_ratio = np.log(df['high'] / df['low']).replace([np.inf, -np.inf], np.nan).fillna(0)

        # Fator constante de Parkinson
        const = 1 / (4 * np.log(2))

        # Volatilidade instantânea
        vol_parkinson = np.sqrt(const * hl_ratio ** 2)

        # Suaviza com janela rolante (opcional)
        vol_rolling = vol_parkinson.rolling(self.window, min_periods=1).mean()

        # Retorna série com mesmo índice
        return pd.Series(vol_rolling, index=df.index)
