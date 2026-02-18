import polars as pl
from Indicator import Indicator

class PriorCote(Indicator):
    def __init__(self, asset=None, timeframe=None, **params):
        super().__init__(asset, timeframe, **params)
        self.name = "prior_quote"

    def _calculate_logic(self, df: pl.DataFrame, **kwargs) -> pl.Series:
        """
        Calcula valores de períodos superiores (ex: Máxima da Semana Anterior).
        Parâmetros esperados em kwargs:
            - col_tf: O timeframe de agrupamento ('W', 'M', 'D', etc.)
            - price_col: A coluna alvo ('high', 'low', 'close', 'open')
        """
        col_tf = kwargs.get('col_tf', '1w').lower() # Default Semana
        if col_tf[0].isalpha():
            col_tf = "1" + col_tf
        price_col = kwargs.get('price_col', 'high')
        
        # Garante que datetime é o índice temporal
        if "datetime" not in df.columns:
            raise ValueError("A coluna 'datetime' é obrigatória para cálculos de Prior Quote.")

        # 1. Agrupamento por período superior
        # 'upsample' ou 'group_by_dynamic' para pegar o valor do período
        
        # Definimos como agregar baseado na coluna
        if price_col == "high":
            agg_expr = pl.col("high").max()
        elif price_col == "low":
            agg_expr = pl.col("low").min()
        elif price_col == "close":
            agg_expr = pl.col("close").last()
        elif price_col == "open":
            agg_expr = pl.col("open").first()
        else:
            agg_expr = pl.col(price_col).last()

        # 2. Criar o DataFrame de períodos superiores
        # Usamos group_by_dynamic para criar as janelas (W, M, etc)
        df_period = (
            df.sort("datetime")
            .group_by_dynamic("datetime", every=col_tf)
            .agg(agg_expr.alias("period_val"))
        )

        # 3. Shift (i-1) para evitar leakage
        # O valor da semana anterior deve estar disponível para a semana atual
        # Exceto se for 'open' no exato momento da abertura, mas por segurança shift(1) é padrão
        df_period = df_period.with_columns(
            pl.col("period_val").shift(1).alias("prior_val")
        )

        # 4. Join de volta para o DataFrame original (D1)
        # O join 'asof' ou join por período garante que cada linha de D1 receba o valor do grupo pai
        
        # Criamos uma coluna temporária no original para o join (início do período)
        df_with_period_start = df.with_columns(
            pl.col("datetime").dt.truncate(col_tf).alias("_period_start")
        )

        result_df = df_with_period_start.join(
            df_period.select(["datetime", "prior_val"]),
            left_on="_period_start",
            right_on="datetime",
            how="left"
        )

        return result_df["prior_val"]