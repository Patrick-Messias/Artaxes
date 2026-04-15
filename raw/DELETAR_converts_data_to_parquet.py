import sys
sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend')
sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend\Assets')
sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend\core')
sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform')
from Asset import Asset, mt5_convert_folder # type: ignore

mt5_convert_folder(
    source_folder="raw",
    asset_type="futures",
    market="b3",
    #datetime_fmt="%Y-%m-%d %H:%M:%S",  # só se a inferência falhar
    # delimiter="\t",                  # só se não for vírgula
    update_reason="initial import from MT5",
)

# assets = Asset.load_all()
# eurusd = assets["WIN$"]
# print(eurusd.data_get("M15"))
















