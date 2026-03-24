import polars as pl
from Indicator import Indicator

class DayOpen(Indicator):
    def __init__(self, asset=None, timeframe=None, **params):
        super().__init__(asset, timeframe, **params)
        self.name = "day_open"

    def _calculate_logic(self, df: pl.DataFrame, **kwargs) -> pl.DataFrame:
        """
        Calcula o preço de abertura do primeiro candle de cada dia e 
        o replica para todos os candles daquele mesmo dia.
        """
        target_name = kwargs.get('ind_name', self.name)
        
        # 1. Criamos um identificador único para o dia (YYYY-MM-DD)
        # Diferente do truncate("1w"), o date() agrupa estritamente por dia civil
        df_work = df.sort("datetime").with_columns(
            pl.col("datetime").dt.date().alias("_day_id")
        )

        # 2. Agrupamos para extrair o PRIMEIRO open de cada dia
        day_stats = (
            df_work.group_by("_day_id", maintain_order=True)
            .agg(
                pl.col("open").first().alias("_first_open")
            )
        )

        # 3. Join de volta: Cada candle (ex: 10:10, 10:20...) se associa ao 
        # _first_open do seu respectivo _day_id.
        # NÃO usamos shift(1) aqui porque o Open do dia é um dado "safe" (não muda)
        result_df = df_work.join(
            day_stats.select(["_day_id", "_first_open"]),
            on="_day_id",
            how="left"
        ).rename({"_first_open": target_name})

        # 4. Limpeza e preenchimento de segurança
        return result_df.select([
            pl.col(target_name).fill_null(strategy="forward").fill_null(0.0)
        ])