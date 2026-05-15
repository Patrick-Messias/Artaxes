import polars as pl
from Indicator import Indicator

class DayOpen(Indicator):
    """
    Calcula o preço de abertura do primeiro candle de cada dia e 
    o replica para todos os candles daquele mesmo dia usando a nova metodologia.
    """
    def __init__(self, asset=None, timeframe=None, **params):
        # Definimos 'price_col' como 'open' por padrão para este indicador
        defaults = {
            'price_col': 'open'
        }
        defaults.update(params)
        super().__init__(asset, timeframe, **defaults)
        self.name = "day_open"

    def _get_expr(self, **kwargs) -> pl.Expr:
        price_col = str(kwargs.get('price_col', 'open'))

        # A "mágica" do Polars para substituir o Join:
        # 1. Pegamos a coluna de preço (open).
        # 2. .first() pega o primeiro valor do grupo.
        # 3. .over(pl.col("datetime").dt.date()) define que o grupo é o dia civil.
        return (
            pl.col(price_col)
            .first()
            .over(pl.col("datetime").dt.date())
            .fill_null(strategy="forward")
            .fill_null(0.0)
        )