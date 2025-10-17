import pandas as pd, itertools

class Indicator:
    def __init__(self, asset, timeframe):
        self.asset = asset
        self.timeframe = timeframe
        #self.data = pd.DataFrame # To be used in future for caching data if needed for backtest
        
    def calculate(self, df): # Abstract method to be implemented by subclasses
        raise NotImplementedError
    
    def calculate_all_sets(self, df, base_path: str = ""):
        import itertools

        param_names, param_values = [], []

        # Identifica atributos que são listas de parâmetros
        for attr, value in self.__dict__.items():
            if attr.startswith('_') or callable(value):
                continue
            if isinstance(value, list):
                param_names.append(attr)
                param_values.append(value)

        results = {}

        # Caso não haja múltiplos parâmetros, calcula direto
        if not param_names:
            key = f"{base_path}.{self._param_suffix()}" if base_path else self._param_suffix()
            results[key] = self.calculate(df)
            return results

        # Gera todas as combinações possíveis
        for combo in itertools.product(*param_values):
            # Salva os valores originais
            original = {name: getattr(self, name) for name in param_names}

            # Atualiza os parâmetros do indicador
            for name, val in zip(param_names, combo):
                setattr(self, name, val)

            # Calcula o resultado
            calculated_data = self.calculate(df)

            # Cria um identificador legível para o conjunto de parâmetros
            param_id = "_".join(f"{name}={val}" for name, val in zip(param_names, combo))

            # Cria o path completo para salvar o resultado
            full_key = f"{base_path}.{param_id}" if base_path else param_id
            results[full_key] = calculated_data

            # Restaura os valores originais
            for name, val in original.items():
                setattr(self, name, val)

        return results

    def _param_suffix(self):
        """Retorna um sufixo compacto com os parâmetros fixos atuais."""
        return "_".join(
            f"{k}={v}"
            for k, v in self.__dict__.items()
            if not k.startswith('_') and not callable(v) and not isinstance(v, list)
        )








""" OLD Indicator.py

    def calculate_all_sets(self, df, base_path: str = ""):
        import itertools

        param_names, param_values = [], []
        
        # Identifica quais atributos são listas de parâmetros
        for attr, value in self.__dict__.items():
            if attr.startswith('_') or callable(value):
                continue
            if isinstance(value, list):
                param_names.append(attr)
                param_values.append(value)
        
        results = {}
        if not param_names:
            key = f"{base_path}.{self._param_suffix()}" if base_path else self._param_suffix()
            results[key] = self.calculate(df)
            return results
        
        # Gera todas as combinações possíveis
        for combo in itertools.product(*param_values):
            # Salva valores originais
            original = {name: getattr(self, name) for name in param_names}

            # Atualiza atributos
            for name, val in zip(param_names, combo):
                setattr(self, name, val)
            
            # Calcula resultado
            calculated_data = self.calculate(df)
            
            # Gera string de parâmetros
            param_id = "_".join(f"{name}{val}" for name, val in zip(param_names, combo))
            full_key = f"{base_path}.{param_id}" if base_path else param_id
            results[full_key] = calculated_data
            
            # Restaura valores originais
            for name, val in original.items():
                setattr(self, name, val)
        
        return results

    def _param_suffix(self):
        # Retorna apenas um sufixo compacto dos parâmetros atuais.
        return ".".join(f"{getattr(self, k)}" for k in self.__dict__ 
                        if not k.startswith('_') and not callable(getattr(self, k)))

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



