import pandas as pd

class Indicator:
    def __init__(self, timeframe):
        self.timeframe = timeframe
        self.data = pd.DataFrame()  # DataFrame to hold calculated indicator values
        
    def calculate(self, df): # Abstract method to be implemented by subclasses
        raise NotImplementedError




""" OLD Indicator.py
@dataclass
class Indicator: # Indicador class utilizado apenas para organizar, os calculos e armazenamento de dados será feito em um dicionário
    name: str # "IND_Var_Parametric"
    timeframe: str
    params: dict # {"alpha": [0.01, 0.05], "sliced_data_length": [50, 100]}

    func_path: Optional[str] = None                # "Indicators.IND_Var_Parametric" ou "TA.RSI"
    sliced_data: bool = False
    sliced_data_length_param: Optional[str] = None # ex: "window", "length", "sliced_data_length"
    input_cols: Optional[List[str]] = None         # ex: ["close"]
    output_name_template: Optional[str] = None     # ex: "{name}[alpha={alpha},len={sliced_data_length}]"


def calculate_indicator(df: pd.DataFrame, ind: Indicator, params: dict) -> pd.DataFrame:
    # Must use indicator name to search for the function to calculate it in files or TA

    func = _resolve_indicator_func(ind)
    if func is None:
        print(f"Indicator not found: {ind.name}")
        return df
    
    input_df = _select_input(df, ind)
    col_name = _build_col_name(ind, params)

    if ind.sliced_data:
        length_param = ind.sliced_data_length_param
        if not length_param or length_param not in params:
            print(f"Parameter window absent {ind.name}")
            return df
        
        win = int(params[length_param])
        out = [np.nan] * len(df)

        for i in range(win - 1, len(df)):
            window = input_df.iloc[i - win + 1:i + 1]
            res = _apply_func(func, window, params)
            if isinstance(res, (pd.Series, list, np.ndarray)): out[i] = pd.Series(res).iloc[-1]
            elif isinstance(res, pd.DataFrame): out[i] = res.iloc[-1, -1] # takes last col/val
            else: out[i] = res
        df[col_name] = out
        return df

    # Full data (rolling window done in func or not existent)
    res = _apply_func(func, input_df, params)
    if isinstance(res, pd.Series): df[col_name] = res.values
    elif isinstance(res, pd.DataFrame):
        for c in res.columns: df[f"{col_name}_{c}"] = res[c].values
    else: df[col_name] = res
    return df
"""



