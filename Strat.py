import pandas as pd, numpy as np, itertools, importlib
from typing import Dict, Optional, Union, List, Callable
from dataclasses import dataclass, field
from BaseClass import BaseClass
from finta import TA
import uuid

from MoneyManager import MoneyManager, MoneyManagerParams
from StratMoneyManager import StratMoneyManager, StratMoneyManagerParams
from Indicator import Indicator

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
    asset_mapping: Union[Asset, Asset_Portfolio]=None #Dict[str, Dict[str, Union[str, List[str]]]] = field(default_factory=dict)

    execution_settings: ExecutionSettings = field(default_factory=ExecutionSettings)
    data_settings: DataSettings = field(default_factory=DataSettings)
    mma_settings: MoneyManagerParams = field(default_factory=MoneyManagerParams) # If mma_rules=None then will use default or PMA or other saved MMA define in Operation. Else it creates a temporary MMA with mma_settings
    time_settings: TimeSettings = field(default_factory=TimeSettings)
    indicators: Dict[str, Indicator] = field(default_factory=dict)

    entry_rules: Dict[str, Callable[[pd.DataFrame], pd.Series]] = field(default_factory=dict)
    tf_exit_rules: Dict[str, Callable[[pd.DataFrame], pd.Series]] = field(default_factory=dict)
    sl_exit_rules: Dict[str, Callable[[pd.DataFrame], pd.Series]] = field(default_factory=dict) 
    tp_exit_rules: Dict[str, Callable[[pd.DataFrame], pd.Series]] = field(default_factory=dict) 
    be_pos_rules: Dict[str, Callable[[pd.DataFrame], pd.Series]] = field(default_factory=dict) 
    be_neg_rules: Dict[str, Callable[[pd.DataFrame], pd.Series]] = field(default_factory=dict) 
    nb_exit_rules: Dict[str, Callable[[pd.DataFrame], pd.Series]] = field(default_factory=dict)

    strat_money_manager: Optional['StratMoneyManager'] = None


class Strat(BaseClass):
    def __init__(self, strat_params: StratParams):
        super().__init__()
        self.name = strat_params.name
        self.asset_mapping = strat_params.asset_mapping

        self.execution_settings = strat_params.execution_settings
        self.data_settings = strat_params.data_settings
        self.mma_settings = strat_params.mma_settings # If mma_rules=None then will use default or PMA or othe MMA define in Operation
        self.time_settings = strat_params.time_settings
        self.indicators = strat_params.indicators

        self.entry_rules = strat_params.entry_rules
        self.tf_exit_rules = strat_params.tf_exit_rules
        self.sl_exit_rules = strat_params.sl_exit_rules
        self.tp_exit_rules = strat_params.tp_exit_rules
        self.be_pos_rules = strat_params.be_pos_rules
        self.be_neg_rules = strat_params.be_neg_rules
        self.nb_exit_rules = strat_params.nb_exit_rules

        # StratMoneyManager is optional - if None, will use default or PMA/MMM from Operation
        self.strat_money_manager = strat_params.strat_money_manager

        self.data = None
    


        
    def _prepare_backtest_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara colunas necessárias para o backtest"""
        # Adiciona colunas padrão se não existirem
        if 'entry_price_long' not in df.columns:
            df['entry_price_long'] = df['close']
        if 'entry_price_short' not in df.columns:
            df['entry_price_short'] = df['close']
            
        return df

    def _apply_entry_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.entry_rules:
            return df
            
        entry_signals = pd.DataFrame(index=df.index)
        
        for rule_name, rule_func in self.entry_rules.items():
            try:
                signal = rule_func(df)
                if isinstance(signal, pd.Series):
                    entry_signals[rule_name] = signal
                else:
                    print(f"Warning: Rule {rule_name} did not return a pandas Series")
            except Exception as e:
                print(f"Error applying entry rule {rule_name}: {str(e)}")
                
        if entry_signals.empty:
            return df
            
        # Combina os sinais (precisa de todos True para entrar)
        df['entry_long'] = entry_signals.all(axis=1)
        df['entry_short'] = entry_signals.all(axis=1)
        
        # Aplica regras de direção - verifica se as regras existem
        if 'entry_long' not in self.entry_rules: df['entry_long'] = False
        if 'entry_short' not in self.entry_rules: df['entry_short'] = False
        
        return df

    def _apply_exit_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        # Trend Following Exit
        if self.tf_exit_rules:
            tf_signals = pd.DataFrame(index=df.index)
            
            for rule_name, rule_func in self.tf_exit_rules.items():
                try:
                    signal = rule_func(df)
                    if isinstance(signal, pd.Series):
                        tf_signals[rule_name] = signal
                    else:
                        print(f"Warning: TF Exit Rule {rule_name} did not return a pandas Series")
                except Exception as e:
                    print(f"Error applying TF exit rule {rule_name}: {str(e)}")
                    
            if not tf_signals.empty:
                # Combina os sinais em um único sinal unificado
                # 1 = compra, -1 = venda, 0 = neutro
                df['tf'] = 0  # Inicializa com neutro
                
                # Se todos os sinais forem True, define como 1 (compra)
                buy_signals = tf_signals.all(axis=1)
                df.loc[buy_signals, 'tf'] = 1
                
                # Se todos os sinais forem False, define como -1 (venda)
                sell_signals = ~tf_signals.any(axis=1)
                df.loc[sell_signals, 'tf'] = -1
        
        # Stop Loss
        if self.sl_exit_rules:
            sl_signals = pd.DataFrame(index=df.index)
            
            for rule_name, rule_func in self.sl_exit_rules.items():
                try:
                    signal = rule_func(df)
                    if isinstance(signal, pd.Series):
                        sl_signals[rule_name] = signal
                    else:
                        print(f"Warning: SL Exit Rule {rule_name} did not return a pandas Series")
                except Exception as e:
                    print(f"Error applying SL exit rule {rule_name}: {str(e)}")
                    
            if not sl_signals.empty:
                df['exit_sl'] = sl_signals.all(axis=1)
        
        return df






    def transfer_HTF_Columns(self, ltf_df: pd.DataFrame, ltf_tf: str, htf_df: pd.DataFrame, htf_tf: str, columns: Optional[List[str]] = None): 
        """
        Transfere colunas do timeframe maior para o menor
        
        Args:
            ltf_df (pd.DataFrame): DataFrame do timeframe menor
            ltf_tf (str): Timeframe menor (ex: 'M5')
            htf_df (pd.DataFrame): DataFrame do timeframe maior
            htf_tf (str): Timeframe maior (ex: 'H1')
            columns (list[str], optional): Lista de colunas para transferir. Se None, transfere todas
            
        Returns:
            pd.DataFrame: DataFrame do timeframe menor com as colunas do maior
        """
        
        def get_tf_minutes(tf: str) -> int:
            """Converte timeframe para minutos"""
            if tf.startswith('M'): return int(tf[1:])
            elif tf.startswith('H'): return int(tf[1:]) * 60
            elif tf.startswith('D'): return int(tf[1:]) * 1440
            else: raise ValueError(f"Timeframe não suportado: {tf}")
            
        if not columns:
            columns = htf_df.columns
            
        ltf_minutes = get_tf_minutes(ltf_tf)
        htf_minutes = get_tf_minutes(htf_tf)
        
        if ltf_minutes >= htf_minutes:
            raise ValueError(f"Timeframe menor ({ltf_tf}) deve ser menor que o maior ({htf_tf})")
            
        # Cria índice de tempo para ambos os DataFrames
        ltf_df = ltf_df.copy()
        htf_df = htf_df.copy()
        
        if 'datetime' not in ltf_df.columns or 'datetime' not in htf_df.columns:
            raise ValueError("Ambos os DataFrames precisam ter coluna 'datetime'")
            
        ltf_df.set_index('datetime', inplace=True)
        htf_df.set_index('datetime', inplace=True)
        
        # Replica valores do HTF para cada barra do LTF
        for column in columns:
            if column in htf_df.columns:
                ltf_df[f"{column}_{htf_tf}"] = htf_df[column].reindex(ltf_df.index, method='ffill')
            
        ltf_df.reset_index(inplace=True)
        return ltf_df





    def __repr__(self):
        return self.__str__()






