import polars as pl
from Indicator import Indicator # type: ignore

class VWAP(Indicator):
    def __init__(self, asset=None, timeframe=None, **params):
        # Parâmetros padrão: vwap diário ancorado na coluna datetime
        defaults = {
            'anchor': 'day',         # Pode ser 'day', 'week', 'month' ou 'none'
            'time_col': 'datetime',  # Coluna de data/hora necessária para agrupar
            'volume_col': 'volume'   # Coluna com o volume das negociações
        }
        defaults.update(params)
        super().__init__(asset, timeframe, **defaults)
        self.name = self.__class__.__name__.lower()

    def _get_expr(self, **kwargs) -> pl.Expr:
        anchor = kwargs.get('anchor', 'day').lower()
        time_col = kwargs.get('time_col', 'datetime')
        volume_col = kwargs.get('volume_col', 'volume')
        
        # 1. Calcula o Preço Típico do candle (Média de High, Low, Close)
        typical_price = (pl.col("high") + pl.col("low") + pl.col("close")) / 3
        
        # 2. Calcula o Preço Típico multiplicado pelo Volume (Preço Ponderado)
        pv = typical_price * pl.col(volume_col)
        
        # 3. Se anchor for 'none', faz o cálculo cumulativo para todo o histórico disponível
        if anchor == 'none':
            return pv.cum_sum() / pl.col(volume_col).cum_sum()
        
        # 4. Caso contrário, define a partição temporal dinamicamente
        if anchor == 'day':
            time_partition = pl.col(time_col).dt.date()
        elif anchor == 'week':
            time_partition = pl.col(time_col).dt.week()
        elif anchor == 'month':
            time_partition = pl.col(time_col).dt.month()
        else:
            raise ValueError(f"Anchor '{anchor}' não suportado. Use 'day', 'week', 'month' ou 'none'.")
            
        # 5. Aplica a soma cumulativa resetando a cada mudança do intervalo de tempo usando .over()
        cum_pv = pv.cum_sum().over(time_partition)
        cum_vol = pl.col(volume_col).cum_sum().over(time_partition)
        
        return cum_pv / cum_vol
