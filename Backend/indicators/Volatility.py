import polars as pl
from Indicator import Indicator # type: ignore

import polars as pl
from Indicator import Indicator # type: ignore

class Volatility(Indicator):
    """
    Calcula a volatilidade (Std Dev) dos retornos para normalização de risco.
    Se aggr_days=True, agrupa retornos intradiários para calcular a vol baseada em dias.
    """
    def __init__(self, asset=None, timeframe=None, **params):
        # window: 21 (um mês comercial)
        # aggr_days: True para normalizar volatilidade diária
        defaults = {
            'window': 21, 
            'aggr_days': True, 
            'price_col': 'pnl_pct',
            'min_periods': 5
        }
        defaults.update(params)
        super().__init__(asset, timeframe, **defaults)
        self.name = self.__class__.__name__.lower()

    def _get_expr(self, **kwargs) -> pl.Expr:
        window = int(kwargs.get('window'))
        aggr_days = kwargs.get('aggr_days')
        price_col = kwargs.get('price_col')
        min_p = kwargs.get('min_periods')

        # Se aggr_days for True, a expressão assume que o motor 
        # enviou dados já agregados por data ou lida com a janela temporal
        if aggr_days:
            # Lógica: Calculamos a volatilidade móvel
            # O Polars trata o rolling_std de forma eficiente.
            # Nota: O retorno aqui deve ser aplicado sobre a coluna de PnL do aggr_ret
            return pl.col(price_col).rolling_std(
                window_size=window, 
                min_periods=min_p
            )
        else:
            # Caso queira volatilidade "tick-to-tick" ou "bar-to-bar" puro
            return pl.col(price_col).rolling_std(window_size=window)
        

