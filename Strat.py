import pandas as pd, numpy as np, itertools, importlib
from typing import Dict, Optional, Union, List, Callable
from dataclasses import dataclass, field
from BaseClass import BaseClass
from Asset import Asset, Asset_Portfolio
from finta import TA
import uuid, inspect

from MoneyManager import MoneyManager, MoneyManagerParams
from StratMoneyManager import StratMoneyManager, StratMoneyManagerParams
from Indicator import Indicator

from Backtest import Backtest, BacktestParams
from Optimization import Optimization
from Walkforward import Walkforward

# Import will be added later to avoid circular import
# from StratMoneyManager import StratMoneyManager

@dataclass
class ExecutionSettings:
    order_type: str='market'
    offset: float=0.0

@dataclass
class TimeSettings:
    day_trade: bool = False
    timeTI: Optional[list[int]] = None
    timeEF: Optional[list[int]] = None
    timeTF: Union[bool, List[int]] = False
    next_index_day_close: bool = False
    friday_close: bool = False
    timeExcludeHours: Optional[List[int]] = None
    dateExcludeTradingDays: Optional[List[int]] = None
    dateExcludeMonths: Optional[List[int]] = None

""" 
@dataclass # # TradeManagementRules Eliminated, it must check in Strat's generate_signals() 
class TradeManagementRules:
    LONG: bool = True
    SHORT: bool = True
    TF: bool = True
    SL: bool = True
    TP: bool = False
    BE_pos: bool = False
    BE_neg: bool = False
    NB: int = -1
"""

@dataclass
class DataSettings: # Commenting dataframe to avoid repetition, additional_timeframes can be replaced by a func
    fill_method: str='ffill'
    fillna: object=0

                                                   #MOVE FUNCTIONS BELOW TO MODEL OR OPERATION WHERE THINGS WILL ACTUALLY HAPPEN

def _generate_param_comb(params: dict) -> list: # Gera as combinações possíveis de parametros
    param_values = []
    param_names = []

    for param_name, param_range in params.items():
        if isinstance(param_range, range): param_values.append(list(param_range))
        elif isinstance(param_range, list): param_values.append(param_range)
        else: param_values.append([param_range])
        param_names.append(param_name)

    combinations = list(itertools.product(*param_values))

    return [dict(zip(param_names, combo)) for combo in combinations]

def _resolve_indicator_func(ind: Indicator):
    # 1 - Explicit func_path
    if ind.func_path:
        mod_name, func_name = ind.func_path.rsplit('.', 1)

        if mod_name == "TA": 
            if hasattr(TA, func_name): return getattr(TA, func_name)
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, func_name): return getattr(mod, func_name)
        except Exception: pass

    # 2 - Local Indicators.py
    try:
        Indicators = importlib.import_module("Indicators")
        if hasattr(Indicators, ind.name): return getattr(Indicators, ind.name)
    except Exception: pass
    
    # 3 - finta.TA
    try:
        if hasattr(TA, ind.name): return getattr(TA, ind.name)
    except Exception: pass

    return None

def _build_col_name(ind: Indicator, params: Dict[str, any]) -> str:
    if ind.output_name_template: return ind.output_name_template.format(name=ind.name, **params)
    suffix = "__".join([f"{k}={params[k]}" for k in sorted(params.keys())])
    return f"{ind.name}__{suffix}"

def _select_input(df: pd.DataFrame, ind: Indicator) -> pd.DataFrame:
    # Para indicadores que precisam de OHLCV (como SMA), retorna o DataFrame completo
    if ind.input_cols and len(ind.input_cols) == 1 and ind.input_cols[0] == "close":
        return df  # Retorna DataFrame completo para indicadores que usam apenas close
    elif ind.input_cols: 
        return df[ind.input_cols].copy()
    return df

def _apply_func(func, data, params):
    try: return func(data, **params)
    except TypeError:
        if isinstance(data, pd.DataFrame) and 'close' in data: 
            return func(data['close'].values, **params)
        return func(np.asarray(data), **params)

def _calculate_indicator(df: pd.DataFrame, ind: Indicator, params: dict) -> pd.DataFrame: # Calculates for one indicator and one parameter
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

                                                   #MOVE FUNCTIONS BELOW TO MODEL OR OPERATION WHERE THINGS WILL ACTUALLY HAPPEN

def calculate_indicators(df: pd.DataFrame, indicators: dict = None) -> pd.DataFrame: # Calculates for all indicators and all parameters
    """
    Calcula indicadores para um DataFrame de dados
    
    Args:
        df: DataFrame com dados OHLCV
        indicators: Dicionário com indicadores para calcular
        
    Returns:
        DataFrame com os indicadores calculados
    """
    if indicators is None: 
        return df
    
    # Calculate Indicators
    for ind_name, ind in indicators.items():
        param_combinations = _generate_param_comb(ind.params)

        for params in param_combinations:
            df = _calculate_indicator(df, ind, params)

    return df


def generate_signals(self, asset_name: str = None, indicators_cache: dict = None) -> pd.DataFrame:
    # Must take DF and indicators, use all model rules to generate when have signals to enter, exit, etc
    return None

@dataclass
class StratParams():
    name: str = field(default_factory=lambda: f'strat_{uuid.uuid4()}')
    operation: Union[Backtest, Optimization, Walkforward]=None
    #strat_support_assets: Optional[Dict[str, Asset]] = field(default_factory=None) #Dict[str, Dict[str, Union[str, List[str]]]] = field(default_factory=dict)

    params: Dict = field(default_factory=dict) 
    execution_settings: ExecutionSettings = field(default_factory=ExecutionSettings)
    data_settings: DataSettings = field(default_factory=DataSettings)
    mma_settings: MoneyManagerParams = field(default_factory=MoneyManagerParams) # If mma_rules=None then will use default or PMA or other saved MMA define in Operation. Else it creates a temporary MMA with mma_settings
    time_settings: TimeSettings = field(default_factory=TimeSettings)
    indicators: Dict[str, Indicator] = field(default_factory=dict) 

    signal_rules: Dict = field(default_factory=lambda: {
        'entry_long': None,
        'entry_short': None,
        'exit_tf_long': None,
        'exit_tf_short': None,
        'exit_sl_long': None,
        'exit_sl_short': None,
        'exit_tp_long': None,
        'exit_tp_short': None,
        'exit_nb_long': None,
        'exit_nb_short': None,
        'be_pos_long': None,
        'be_pos_short': None,
        'be_neg_long': None,
        'be_neg_short': None
    })

    strat_money_manager: Optional['StratMoneyManager'] = None

def call_rule_function(func, **kwargs):
    """Chama uma função com apenas os argumentos que ela realmente usa."""
    sig = inspect.signature(func)
    valid_args = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(**valid_args)
    
class Strat(BaseClass):
    def __init__(self, strat_params: StratParams):
        super().__init__()
        self.name = strat_params.name
        self.operation = strat_params.operation
        #self.strat_support_assets = strat_params.strat_support_assets

        self.params = strat_params.params
        self.execution_settings = strat_params.execution_settings
        self.data_settings = strat_params.data_settings
        self.mma_settings = strat_params.mma_settings # If mma_rules=None then will use default or PMA or othe MMA define in Operation
        self.time_settings = strat_params.time_settings
        self.indicators = strat_params.indicators

        self.signal_rules = strat_params.signal_rules

        # self.entry_rules = strat_params.entry_rules
        # self.tf_exit_rules = strat_params.tf_exit_rules
        # self.sl_exit_rules = strat_params.sl_exit_rules
        # self.tp_exit_rules = strat_params.tp_exit_rules
        # self.be_pos_rules = strat_params.be_pos_rules
        # self.be_neg_rules = strat_params.be_neg_rules
        # self.nb_exit_rules = strat_params.nb_exit_rules

        # StratMoneyManager is optional - if None, will use default or PMA/MMM from Operation
        self.strat_money_manager = strat_params.strat_money_manager

        self.data = None
    
    def generate_signals(self, df, ind_series_dict, param_id, params=None):
        results = {
            'param_id': param_id,
            'signals_long':  call_rule_function(
                self.entry_rules.get('entry_long', lambda df, inds: None),
                df=df, ind_series_dict=ind_series_dict, param_id=param_id, params=params
            ),
            'signals_short': call_rule_function(
                self.entry_rules.get('entry_short', lambda df, inds: None),
                df=df, ind_series_dict=ind_series_dict, param_id=param_id, params=params
            ),
            'exit_rules': {}
        }

        for rule_group_name in ['tf_exit_rules', 'sl_exit_rules', 'tp_exit_rules', 'be_pos_rules', 'be_neg_rules']:
            rule_group = getattr(self, rule_group_name, {}) or {}
            for side in ['long', 'short']:
                key = f"{rule_group_name}_{side}"
                if f"{side}" in [k.split('_')[-1] for k in rule_group.keys()]:
                    func_key = next(k for k in rule_group if k.endswith(side))
                    results['exit_rules'][key] = call_rule_function(
                        rule_group[func_key],
                        df=df, ind_series_dict=ind_series_dict, param_id=param_id, params=params
                    )

        return results


    def _prepare_backtest_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara colunas necessárias para o backtest"""
        # Adiciona colunas padrão se não existirem
        if 'entry_price_long' not in df.columns:
            df['entry_price_long'] = df['close']
        if 'entry_price_short' not in df.columns:
            df['entry_price_short'] = df['close']
            
        return df












    def __repr__(self):
        return self.__str__()






