import sys
sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend')
sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend\Assets')
sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend\core')
sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform')
from Asset import Asset, convert_folder # type: ignore

# convert_folder(
#     source_folder="raw/Forex",
#     asset_type="currency_pair",
#     market="forex",
#     # datetime_fmt="%Y.%m.%d %H:%M",  # só se a inferência falhar
#     # delimiter="\t",                  # só se não for vírgula
#     update_reason="initial import from MT5",
# )

assets = Asset.load_all()
eurusd = assets["EURUSD"]
print(eurusd.data_get("M15"))
















