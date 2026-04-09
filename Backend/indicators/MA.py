import polars as pl
from Indicator import Indicator # type: ignore

class MA(Indicator):
    def __init__(self, asset=None, timeframe=None, **params):
        # Default params if not provided
        defaults = {'window': 21, 'ma_type': 'sma', 'price_col': 'close'}
        defaults.update(params)
        super().__init__(asset, timeframe, **defaults)
        self.name = self.__class__.__name__.lower()

    def _get_expr(self, **kwargs) -> pl.Expr:
        window = int(kwargs.get('window'))
        ma_type = kwargs.get('ma_type', 'sma').lower()
        price_col = kwargs.get('price_col', 'close')

        # Logic is pure Polars API - No .to_series() or .select() here!
        if ma_type == 'sma':
            return pl.col(price_col).rolling_mean(window_size=window)
        elif ma_type == 'ema':
            return pl.col(price_col).ewm_mean(span=window, adjust=False)
        
        raise ValueError(f"Unsupported MA type: {ma_type}")


"""
class MA(Indicator):
    def __init__(self, asset=None, timeframe=None, **params):
        super().__init__(asset, timeframe, **params)
        self.name = "ma"

    def _calculate_logic(self, data, **kwargs) -> pl.Series:
        window = int(kwargs.get('window', 21))
        ma_type = kwargs.get('ma_type', 'sma')
        price_col = kwargs.get('price_col', 'close')

        # Input treatment (Series/DataFrame)
        if isinstance(data, pl.Series): df = data
        elif isinstance(data, pl.DataFrame):
            if price_col not in data.columns:
                actual_col = next((c for c in data.columns if c.lower() not in ['ts', 'datetime']), data.columns[0])
                price_col = actual_col
            df = data.get_column(price_col)
        else:
            raise ValueError(f"MA: Type of data {type(data)} not supported, use pl.Series/DataFrame")

        if ma_type == 'sma':
            return df.rolling_mean(window_size=window)
        elif ma_type == 'ema':
            return df.to_frame().select(
                pl.col(df.name).ewm_mean(span=window, adjust=False)
            ).to_series()
        raise ValueError(f"Unsupported MA type: {ma_type}")
"""



