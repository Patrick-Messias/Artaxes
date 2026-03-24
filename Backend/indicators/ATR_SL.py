import polars as pl
from Indicator import Indicator

class ATR_SL(Indicator):
    def __init__(self, asset=None, timeframe=None, **params):
        super().__init__(asset, timeframe, **params)
        self.name = "atr"

    def _calculate_logic(self, df: pl.DataFrame, **kwargs) -> pl.Series:
        window = int(kwargs.get('window', 21))
        high_col = kwargs.get('high_col', 'high')
        low_col = kwargs.get('low_col', 'low')

        # Ajuste automático de nome de coluna (case-insensitive)
        for col_name, var_name in [(high_col, 'high_col'), (low_col, 'low_col')]:
            if col_name not in df.columns:
                actual_col = next((c for c in df.columns if c.lower() == col_name.lower()), None)
                if actual_col:
                    if var_name == 'high_col':
                        high_col = actual_col
                    else:
                        low_col = actual_col
                else:
                    raise ValueError(f"Coluna '{col_name}' não encontrada. Colunas disponíveis: {df.columns}")

        return (
            df.select(
                (pl.col(high_col) - pl.col(low_col))
                .rolling_mean(window_size=window)
                .fill_null(0)
            )
            .to_series()
        )
