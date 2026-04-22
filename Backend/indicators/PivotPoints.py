import polars as pl
from Backend.core.Indicator import Indicator 

# Pivot Points with no lookahead bias. The pivot point is the highest high or lowest low within a specified lookback radius.

class PivotPoints(Indicator):
    def __init__(self, asset=None, timeframe=None, **params):
        # Default params if not provided
        defaults = {'lookback_radius': 3, 'price_col_hi': 'high', 'price_col_lo': 'low'}
        defaults.update(params)
        super().__init__(asset, timeframe, **defaults)
        self.name = self.__class__.__name__.lower()

    def _get_expr(self, **kwargs) -> pl.Expr:
        lookback_radius = int(kwargs.get('lookback_radius', 3))
        price_col_hi = kwargs.get('price_col_hi', 'high')
        price_col_lo = kwargs.get('price_col_lo', 'low')
        perfect_pivot = kwargs.get('perfect_pivot', False)
        window = (lookback_radius * 2) + 1
        
        hi = pl.col(price_col_hi)
        lo = pl.col(price_col_lo)

        pivot_hi = hi == hi.rolling_max(window_size=window, center=True)
        pivot_lo = lo == lo.rolling_min(window_size=window, center=True)

        return [
            pl.when(pivot_hi).then(hi).otherwise(None).shift(lookback_radius).alias('pivot_high'),
            pl.when(pivot_lo).then(lo).otherwise(None).shift(lookback_radius).alias('pivot_low')
        ]




