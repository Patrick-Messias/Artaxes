import sys
sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend')
sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend\Assets')
sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend\core')
sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform')
from Asset import Asset, mt5_convert_folder # type: ignore

# mt5_convert_folder(
#     source_folder="raw",
#     asset_type="stock",
#     market="b3",
#     submarket="generic",
#     #datetime_fmt="%Y-%m-%d %H:%M:%S",  # só se a inferência falhar
#     # delimiter="\t",                  # só se não for vírgula
#     update_reason="initial import from MT5",
# )

assets = Asset.load_all()
win = assets["EURUSD"]
higherDF = win.data_get("D1")
lowerDF = win.data_get("M15")

print(lowerDF)



# # Quick data integrity test

# import polars as pl, matplotlib.pyplot as plt

# df_m10_aggr = lowerDF.group_by_dynamic(
#     "datetime", every="1d"
# ).agg(
#     pl.col("open").first().alias("open_m10"),
#     pl.col("high").max().alias("high_m10"),
#     pl.col("low").min().alias("low_m10"),
#     pl.col("close").last().alias("close_m10"),
#     pl.col("volume").sum().alias("volume_m10")
# )

# df_analise = higherDF.join(df_m10_aggr, on="datetime", how="inner")

# df_analise = df_analise.with_columns(
#     (pl.col("open") - pl.col("open_m10")).abs().alias("err_open"),
#     (pl.col("high") - pl.col("high_m10")).abs().alias("err_high"),
#     (pl.col("low") - pl.col("low_m10")).abs().alias("err_low"),
#     (pl.col("close") - pl.col("close_m10")).abs().alias("err_close"),
#     (pl.col("volume") - pl.col("volume_m10")).abs().alias("err_volume")
# )

# # Criar painel com 2 subplots (Preços vs Volumes)
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# # Gráfico 1: Erros nos Preços (OHLC)
# ax1.plot(df_analise["datetime"], df_analise["err_open"], label="Erro Open", alpha=0.7)
# ax1.plot(df_analise["datetime"], df_analise["err_high"], label="Erro High", alpha=0.7)
# ax1.plot(df_analise["datetime"], df_analise["err_low"], label="Erro Low", alpha=0.7)
# ax1.plot(df_analise["datetime"], df_analise["err_close"], label="Erro Close", alpha=0.7)
# ax1.set_ylabel("Diferença Absoluta (Pontos)")
# ax1.set_title("Inconsistências Detectadas: Diário vs M10 Agregado")
# ax1.legend(loc="upper left")
# ax1.grid(True, linestyle="--", alpha=0.5)

# # Gráfico 2: Erros no Volume
# ax2.plot(df_analise["datetime"], df_analise["err_volume"], label="Erro Volume", color="purple", alpha=0.7)
# ax2.set_ylabel("Diferença Absoluta (Contratos)")
# ax2.set_xlabel("Data")
# ax2.legend(loc="upper left")
# ax2.grid(True, linestyle="--", alpha=0.5)

# plt.tight_layout()
# plt.show()

















