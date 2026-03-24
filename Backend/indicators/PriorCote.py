import polars as pl
from Indicator import Indicator

class PriorCote(Indicator):
    def __init__(self, asset=None, timeframe=None, **params):
        super().__init__(asset, timeframe, **params)
        self.name = "prior_quote"

    def _calculate_logic(self, df: pl.DataFrame, **kwargs) -> pl.DataFrame:
        target_name = kwargs.get('ind_name', self.name)
        price_col = kwargs.get('price_col', 'high')
        
        # 1. Truncamos a data para o início da semana (Sempre Segunda-feira 00:00)
        # Isso cria um identificador único e consistente para a semana
        df_work = df.sort("datetime").with_columns(
            pl.col("datetime").dt.truncate("1w").alias("_week_start")
        )

        # 2. Agrupamos para achar o valor final daquela semana
        # O group_by comum aqui é melhor que o dynamic para garantir o valor por 'balde' de semana
        agg_map = {
            "high": pl.col("high").max(),
            "low": pl.col("low").min(),
            "close": pl.col("close").last(),
            "open": pl.col("open").first()
        }
        agg_expr = agg_map.get(price_col, pl.col(price_col).last())

        week_stats = (
            df_work.group_by("_week_start", maintain_order=True)
            .agg(agg_expr.alias("_final_val"))
            .sort("_week_start")
        )

        # 3. SHIFT(1): O valor da semana que começou em 04/02 
        # agora será atribuído à semana que começa em 11/02
        week_stats = week_stats.with_columns(
            pl.col("_final_val").shift(1).alias(target_name)
        )

        # 4. Join de volta: Cada candle de 10min da semana X recebe o valor da semana X-1
        result_df = df_work.join(
            week_stats.select(["_week_start", target_name]),
            on="_week_start",
            how="left"
        )

        # 5. Preenchimento de segurança para o início do histórico
        return result_df.select([
            pl.col(target_name).fill_null(strategy="forward").fill_null(0.0)
        ])

        
