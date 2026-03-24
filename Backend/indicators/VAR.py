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

    def _calculate_logic(self, df: pl.DataFrame, **kwargs) -> pl.Series:
        window    = int(kwargs.get('window',    20))
        alpha     = float(kwargs.get('alpha',   0.05))
        var_type  = str(kwargs.get('var_type',  'historical')).lower()
        price_col = str(kwargs.get('price_col', 'close'))

        if price_col not in df.columns:
            actual_col = next((c for c in df.columns if c.lower() == price_col.lower()), None)
            if actual_col:
                price_col = actual_col
            else:
                raise ValueError(f"Coluna '{price_col}' não encontrada. Colunas: {df.columns}")

        # Calcula pct_change (retornos) se price_col for 'close' ou similar
        returns = df.select(
            pl.col(price_col).pct_change().fill_null(0.0)
        ).to_series()

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
            raise ValueError(f"var_type não suportado: '{var_type}'. Use 'historical' ou 'parametric'.")

        return var_series.fill_null(0.0).alias("var")