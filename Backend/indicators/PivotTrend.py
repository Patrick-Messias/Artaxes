import polars as pl
import numpy as np
from typing import Union, List, Dict, Any

class PivotTrend(Indicator):
    """
    Classifies trend using the last N pivot highs and lows.
    
    Parameters
    ----------
    window : int, default 63
        Lookback window in periods. Pivots older than (current_index - window) are ignored.
        If -1, all pivots up to the current bar are considered.
    pivots : int, default 3
        Number of most recent pivot highs and lows to keep in the FIFO queue.
    lookback_radius : int, default 3
        Radius for local maxima/minima detection (passed to PivotPoints).
    price_col_hi : str, default 'high'
        Column name for high prices.
    price_col_lo : str, default 'low'
        Column name for low prices.
    
    Returns
    -------
    pl.Expr
        A Polars expression that evaluates to an integer column:
        1  = uptrend (highs and lows are both rising)
        -1 = downtrend (highs and lows are both falling)
        0  = neutral / sideways / insufficient data
    """
    
    def __init__(self, asset: str = None, timeframe: str = None, **params):
        defaults = {
            'window': 63,
            'pivots': 3,
            'lookback_radius': 3,
            'price_col_hi': 'high',
            'price_col_lo': 'low'
        }
        defaults.update(params)
        super().__init__(asset, timeframe, **defaults)
        self.name = 'pivottrend'
    
    def _get_expr(self, **kwargs) -> pl.Expr:
        window = kwargs.get('window', 63)
        pivots_count = kwargs.get('pivots', 3)
        lookback_radius = kwargs.get('lookback_radius', 3)
        price_col_hi = kwargs.get('price_col_hi', 'high')
        price_col_lo = kwargs.get('price_col_lo', 'low')
        
        # Generate pivot point expressions
        pp = PivotPoints(
            asset=self.asset,
            timeframe=self.timeframe,
            lookback_radius=lookback_radius,
            price_col_hi=price_col_hi,
            price_col_lo=price_col_lo
        )
        pivot_exprs = pp._get_expr(**kwargs)  # [pivot_high_expr, pivot_low_expr]
        pivot_high_expr = pivot_exprs[0].alias('pivot_high')
        pivot_low_expr = pivot_exprs[1].alias('pivot_low')
        
        # Row index for time‑based filtering
        idx_expr = pl.int_range(pl.len()).alias('__idx')
        
        # Combine into a struct column for batched UDF processing
        struct_expr = pl.struct([
            pivot_high_expr,
            pivot_low_expr,
            idx_expr
        ])
        
        # Python UDF that computes the trend statefully over the whole series
        def compute_trend(s: pl.Series) -> pl.Series:
            # Unnest the struct into three columns
            df = s.struct.unnest()
            high_vals = df['pivot_high'].to_numpy()
            low_vals = df['pivot_low'].to_numpy()
            indices = df['__idx'].to_numpy()
            
            n = len(high_vals)
            trend = np.zeros(n, dtype=np.int8)
            
            high_queue = []   # list of (idx, value)
            low_queue = []    # list of (idx, value)
            
            for i in range(n):
                # Add new pivots if present
                if not np.isnan(high_vals[i]):
                    high_queue.append((indices[i], high_vals[i]))
                if not np.isnan(low_vals[i]):
                    low_queue.append((indices[i], low_vals[i]))
                
                # Remove expired pivots outside the window
                if window != -1:
                    cutoff = indices[i] - window
                    high_queue = [x for x in high_queue if x[0] >= cutoff]
                    low_queue = [x for x in low_queue if x[0] >= cutoff]
                
                # Keep only the most recent 'pivots_count' entries (FIFO)
                high_queue = high_queue[-pivots_count:]
                low_queue = low_queue[-pivots_count:]
                
                # Determine trend if we have at least two points in each queue
                if len(high_queue) >= 2 and len(low_queue) >= 2:
                    high_increasing = all(
                        high_queue[j][1] < high_queue[j+1][1]
                        for j in range(len(high_queue)-1)
                    )
                    high_decreasing = all(
                        high_queue[j][1] > high_queue[j+1][1]
                        for j in range(len(high_queue)-1)
                    )
                    low_increasing = all(
                        low_queue[j][1] < low_queue[j+1][1]
                        for j in range(len(low_queue)-1)
                    )
                    low_decreasing = all(
                        low_queue[j][1] > low_queue[j+1][1]
                        for j in range(len(low_queue)-1)
                    )
                    
                    if high_increasing and low_increasing:
                        trend[i] = 1
                    elif high_decreasing and low_decreasing:
                        trend[i] = -1
                    else:
                        trend[i] = 0
                else:
                    trend[i] = 0
                    
            return pl.Series(trend)
        
        # Apply the UDF to the struct column
        trend_expr = struct_expr.map_batches(
            compute_trend,
            return_dtype=pl.Int8
        )
        
        return trend_expr


        