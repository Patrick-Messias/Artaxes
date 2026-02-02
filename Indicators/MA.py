import polars as pl
from Indicator import Indicator

class MA(Indicator):
    def _calculate_logic(self, df: pl.DataFrame, **kwargs) -> pl.Series:
        window = int(kwargs.get('window', 21))
        ma_type = kwargs.get('ma_type', 'sma')
        price_col = kwargs.get('price_col', 'close')

        if price_col not in df.columns:
            actual_col = next((c for c in df.columns if c.lower() == price_col.lower()), None)
            if actual_col:
                price_col = actual_col
            else:
                raise ValueError(f"Coluna '{price_col}' não encontrada. Colunas disponíveis: {df.columns}")

        if ma_type == 'sma':
            return df.select(pl.col(price_col).rolling_mean(window_size=window)).to_series()
        elif ma_type == 'ema':
            return df.select(pl.col(price_col).ewm_mean(span=window, adjust=False)).to_series()
        raise ValueError(f"Unsupported MA type: {ma_type}")

