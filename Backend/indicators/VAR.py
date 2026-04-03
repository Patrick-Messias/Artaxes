import polars as pl
import numpy as np
from Indicator import Indicator


class VAR(Indicator):
    """
    Value at Risk (VaR) rolante baseado em retornos do ativo.

    Métodos:
        "parametric"  → VaR = mean - z_score × std  (distribuição normal)
        "historical"  → VaR = percentil alpha dos retornos históricos

    Retorna série de VaR (valores negativos = perda esperada).
    Útil como proxy de volatilidade para position sizing — lote = risk_pct / |VaR|.
    """

    def __init__(self, asset=None, timeframe=None, **params):
        super().__init__(asset, timeframe, **params)
        self.name = "var"

    def _calculate_logic(self, data, **kwargs) -> pl.Series:
        window    = int(kwargs.get('window',    20))
        alpha     = float(kwargs.get('alpha',   0.05))
        var_type  = str(kwargs.get('var_type',  'historical')).lower()
        price_col = str(kwargs.get('price_col', 'close'))

        # Series extraction
        if isinstance(data, pl.Series):
            df = data
            is_returns = True
        elif isinstance(data, pl.DataFrame):
            if price_col not in data.columns:
                price_col = next((c for c in data.columns if c.lower() not in ['ts', 'datetime']), data.columns[0])
            df = data.get_column(price_col)
            is_returns = False
        else:
            raise ValueError(f"VAR: Type not supported: {type(data)}")

        # Calculates pct_change if only brute price
        returns = df if is_returns else df.pct_change().fill_null(0.0)

        if var_type == 'historical': # rolling_quantile nativo — sem overhead Python por janela
            var_series = returns.rolling_quantile(
                quantile=alpha,
                interpolation='linear',
                window_size=window,
            )
        elif var_type == 'parametric':
            from scipy.stats import norm
            z_score = float(norm.ppf(1 - alpha))

            rolling_mean = returns.rolling_mean(window_size=window)
            rolling_std  = returns.rolling_std(window_size=window, ddof=1)

            # Operação direta entre Series — sem pl.lit()
            var_series = rolling_mean - (rolling_std * z_score)

        else:
            raise ValueError(f"Type of VaR not supported: '{var_type}'. Use 'historical' or 'parametric'.")

        return var_series.fill_null(0.0).alias("var")