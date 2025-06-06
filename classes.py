import pandas as pd
import numpy as np
import os
import sys
import re
import itertools
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Union, Callable, Optional, Any, Type

def load_data(data_path, min_limit=0, max_limit=1, index_col_setting=False, drop_index=True):
    """
    Carrega dados de um arquivo ou diretório, tentando automaticamente extensões .xlsx, .xls e .csv.
    """
    def normalize_columns(df):
        if any(col != col.lower() for col in df.columns):
            df.columns = df.columns.str.lower()
        return df

    def process_file(file_path):
        try:
            if file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path, index_col=index_col_setting)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                return None
                
            df = normalize_columns(pd.DataFrame(df))
            df['ativo'] = os.path.splitext(os.path.basename(file_path))[0]
            
            if 'datetime' not in df.columns:
                if 'date' in df.columns: 
                    time_part = (' ' + df['time'].astype(str)) if 'time' in df.columns else '00:00:00'
                    df['datetime'] = pd.to_datetime(df['date'].astype(str) + time_part)
                elif 'time' in df.columns: 
                    df['datetime'] = pd.to_datetime(df['time'].astype(str) + ' 00:00:00')
            
            df['time'] = df.get('time', pd.to_datetime(df['datetime']).dt.time if 'datetime' in df.columns else '00:00:00')
            df['date'] = df.get('date', pd.to_datetime(df['datetime']).dt.date if 'datetime' in df.columns else "1900-00-00")
            
            return df.iloc[int(len(df)*min_limit):int(len(df)*max_limit)].reset_index(drop=drop_index)
        except Exception as e:
            print(f"Erro ao processar {file_path}: {str(e)}")
            return None

    if os.path.isdir(data_path):
        files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
                if f.endswith(('.xlsx', '.xls', '.csv'))]
        results = [process_file(f) for f in files]
        return [df for df in results if df is not None]
    
    possible_extensions = ['.xlsx', '.xls', '.csv']
    
    if any(data_path.endswith(ext) for ext in possible_extensions):
        result = process_file(data_path)
        return result if result is not None else pd.DataFrame()
    
    for ext in possible_extensions:
        file_path = data_path + ext
        if os.path.exists(file_path):
            result = process_file(file_path)
            if result is not None:
                return result
    
    print(f"Nenhum arquivo encontrado para: {data_path} com extensões {possible_extensions}")
    return pd.DataFrame()

class BaseClass():
    def get_value(self, attr_value: str=None):
        if attr_value is None: return list(self.__dict__.keys())
        elif hasattr(self, attr_value): return getattr(self, attr_value)
        else: raise AttributeError(f"!!! --- Attribute '{attr_value}' not found in instance --- !!!")

    def delete_value(self, attr_value: str=None):
        if attr_value is None: return list(self.__dict__.keys())
        elif hasattr(self, attr_value): delattr(self, attr_value) 
        else: raise AttributeError(f"!!! --- Attribute '{attr_value}' not found in instance --- !!!")

    def modify_specific_value(self, attr_value: str, new_attr_value):
        if attr_value is None: return list(self.__dict__.keys())
        elif hasattr(self, attr_value): setattr(self, attr_value, new_attr_value)
        else: raise AttributeError(f"!!! --- Attribute '{attr_value}' not found in instance --- !!!")

    def list_values(self): return {key: value for key, value in self.__dict__.items()} 

# =================================================================================================================================|| ASSET

@dataclass
class AssetParams:
    tick: float = 0.01
    tick_fin_val: float = 1.0
    lot_value: float = 100.0
    min_lot: float = 1.0
    leverage: float = 1.0
    comissions: float = 0.0
    slippage: float = 0.0
    spread: float = 0.0

class Asset(BaseClass):
    ASSET_PARAMS = {
        'futures': {
            'b3': {
                'WIN$': {
                    'tick': 1,
                    'tick_fin_val': 0.2,
                    'lot_value': 20000.0,
                    'min_lot': 1,
                    'leverage': 20,
                    'comissions': 0.5,
                    'slippage': 0.25,
                    'spread': 0.25
                },
                'WDO$': {
                    'tick': 5,
                    'tick_fin_val': 5.0,
                    'lot_value': 40000.0,
                    'min_lot': 1,
                    'leverage': 20,
                    'comissions': 0.5,
                    'slippage': 0.25,
                    'spread': 0.25
                }
            }
        },
        'currency_pair': {
            'forex': {
                'generic': {
                    'tick': 0.0001,
                    'tick_fin_val': 10,
                    'lot_value': 100000.0,
                    'min_lot': 0.01,
                    'leverage': 100,
                    'comissions': 1.5,
                    'slippage': 0.75,
                    'spread': 0.75
                }
            }
        },
        'stock': {
            'NASDAQ': {
                'generic': {
                    'tick': 0.01,
                    'tick_fin_val': 1,
                    'lot_value': 100,
                    'min_lot': 1,
                    'leverage': 1,
                    'comissions': 5.0,
                    'slippage': 0.05,
                    'spread': 0.02
                }
            },
            'NYSE': {
                'generic': {
                    'tick': 0.01,
                    'tick_fin_val': 1,
                    'lot_value': 100,
                    'min_lot': 1,
                    'leverage': 1,
                    'comissions': 5.0,
                    'slippage': 0.05,
                    'spread': 0.02
                }
            }
        }
    }

    def __init__(self, 
                 name: str, 
                 type: str, 
                 market: str,
                 timeframe: list[str] = None,
                 data_path: str = 'C:\\Users\\Patrick\\Desktop\\Artaxes Portfolio\\MAIN\\MT5_Dados',
                 **kwargs
                 ):
        super().__init__()
        if not isinstance(name, str) or not name: raise ValueError("Nome do ativo inválido")
        if type not in self.ASSET_PARAMS: raise ValueError(f"Tipo de ativo inválido. Opções: {list(self.ASSET_PARAMS.keys())}")
        
        self.name = name
        self.type = type
        self.market = market
        self.data_path = data_path
        self.data: dict[str, pd.DataFrame] = {}
        if timeframe: 
            for tf in timeframe:
                self.data[tf] = None
        else: self.timeframes_load_available()

        if not os.path.isdir(self.data_path): print(f"⚠️ Aviso: Caminho de dados '{self.data_path}' não encontrado ao inicializar Asset.")

        default_params = self.get_default_params()
        for key, value in {**default_params, **kwargs}.items():
            setattr(self, key, value)

    def get_default_params(self):
        try:
            return self.ASSET_PARAMS[self.type][self.market][self.name].copy()
        except KeyError:
            try:
                return self.ASSET_PARAMS[self.type][self.market]['generic'].copy()
            except KeyError:
                print(f"!!! --- Warning, default parameter configuration not found for {self.type}/{self.market}, using generic parameters --- !!!")
            return { 
                'tick': 0.01,
                'tick_fin_val': 1.0,
                'lot_value': 100.0,
                'min_lot': 1.0,
                'leverage': 1.0,
                'comissions': 0.0,
                'slippage': 0.0,
                'spread': 0.0
            }

    @staticmethod
    def register_asset_params(name: str, type: str, market: str, params: dict):
        required_params = {'tick', 'tick_fin_val', 'lot_value', 'min_lot', 'leverage', 'comissions', 'slippage', 'spread'}

        missing_params = required_params - set(params.keys())
        if missing_params: raise ValueError(f"!!! --- Necessary Parameters Missing: {missing_params} --- !!!")
        
        if type not in Asset.ASSET_PARAMS: Asset.ASSET_PARAMS[type] = {}
        if market not in Asset.ASSET_PARAMS[type]: Asset.ASSET_PARAMS[type][market] = {}

        Asset.ASSET_PARAMS[type][market][name] = params.copy()
        print(f"✅ Parameters Registered Sucessfully for {market}/{type}/{name}")

    def data_add(self, timeframe: str, data: pd.DataFrame):
        self.data[timeframe] = data

    def data_refresh(self, timeframe: str):
        self.data[timeframe] = None

    def data_is_empty(self, timeframe: str) -> bool:
        return timeframe not in self.data or self.data[timeframe] is None or len(self.data[timeframe]) == 0

    def data_get(self, timeframe: str) -> pd.DataFrame:
        if self.data_is_empty(timeframe):
            data_file = os.path.join(self.data_path, f"{self.name}_{timeframe}")
            self.data[timeframe] = load_data(data_file)
        return self.data[timeframe]

    def timeframes_list(self) -> list[str]:
        return list(self.data.keys())

    def timeframes_load_available(self):
        if not os.path.isdir(self.data_path): return
        
        pattern = re.compile(f"{self.name}_([A-Z][0-9]+)")
        for file in os.listdir(self.data_path):
            match = pattern.match(file)
            if match:
                timeframe = match.group(1)
                if timeframe not in self.data:
                    self.data[timeframe] = None

    @staticmethod
    def load_unique_assets(assets: dict[str, "Asset"], type: str, market: str, path: str) -> dict[str, "Asset"]:
        if not os.path.isdir(path): return assets
        
        pattern = re.compile(r"([A-Za-z0-9$]+)_[A-Z][0-9]+\.(csv|xlsx|xls)")
        for file in os.listdir(path):
            match = pattern.match(file)
            if match:
                asset_name = match.group(1)
                if asset_name not in assets:
                    assets[asset_name] = Asset(asset_name, type, market, data_path=path)
        return assets

    @staticmethod
    def predefined_assets() -> dict[str, "Asset"]:
        assets = {}
        
        # B3 Futures
        b3_path = "C:\\Users\\Patrick\\Desktop\\Artaxes Portfolio\\MAIN\\MT5_Dados\\B3"
        assets = Asset.load_unique_assets(assets, "futures", "b3", b3_path)
        
        # Forex
        forex_path = "C:\\Users\\Patrick\\Desktop\\Artaxes Portfolio\\MAIN\\MT5_Dados\\FOREX"
        assets = Asset.load_unique_assets(assets, "currency_pair", "forex", forex_path)
        
        return assets 

class Asset_Portfolio(BaseClass):
    def __init__(self, asset_portfolio_params: dict):
        super().__init__()
        self.name = asset_portfolio_params.get('name', 'unnamed_portfolio')
        self.assets: dict[str, Asset] = {}
        assets_dict = asset_portfolio_params.get('assets', {})
        if isinstance(assets_dict, dict):
            self.assets = assets_dict
        elif isinstance(assets_dict, list):  # For backward compatibility
            for asset in assets_dict:
                if isinstance(asset, Asset):
                    self.assets[asset.name] = asset

    def __str__(self):
        assets_str = ", ".join(self.assets.keys())
        return f"Portfolio '{self.name}' with assets: {assets_str}"

    def asset_add(self, asset: Asset):
        self.assets[asset.name] = asset

    def asset_remove(self, asset_name: str):
        if asset_name in self.assets:
            del self.assets[asset_name]

    def assets_filter(self, condition: callable) -> list[Asset]:
        return [asset for asset in self.assets.values() if condition(asset)]

    def asset_get(self, name: str) -> Asset:
        if name not in self.assets:
            raise ValueError(f"Asset '{name}' not found in portfolio")
        return self.assets[name]

    def assets_list(self, print_assets: bool = True, sort_by: str = None) -> list:
        assets_info = []
        for asset in self.assets.values():
            info = {
                'name': asset.name,
                'type': asset.type,
                'market': asset.market
            }
            assets_info.append(info)
            
        if sort_by and sort_by in assets_info[0]:
            assets_info.sort(key=lambda x: x[sort_by])
            
        if print_assets:
            for info in assets_info:
                print(f"Asset: {info['name']}, Type: {info['type']}, Market: {info['market']}")
                
        return assets_info

    def data_get(self, timeframes: list[str]) -> dict[str, dict[str, pd.DataFrame]]:
        result = {}
        for timeframe in timeframes:
            result[timeframe] = {}
            for asset_name, asset in self.assets.items():
                try:
                    result[timeframe][asset_name] = asset.data_get(timeframe)
                except Exception as e:
                    print(f"Error loading data for {asset_name} at {timeframe}: {str(e)}")
                    result[timeframe][asset_name] = pd.DataFrame()
        return result

    def stats(self) -> dict:
        return {
            'total_assets': len(self.assets),
            'asset_types': list(set(asset.type for asset in self.assets.values())),
            'markets': list(set(asset.market for asset in self.assets.values()))
        }

    def calculate_correlation(self, timeframe: str, method='pearson') -> pd.DataFrame:
        data = {}
        for asset_name, asset in self.assets.items():
            df = asset.data_get(timeframe)
            if not df.empty and 'close' in df.columns:
                data[asset_name] = df['close']
            
        if not data:
            print("No data available for correlation calculation")
            return pd.DataFrame()
            
        df_combined = pd.DataFrame(data)
        return df_combined.corr(method=method)

    def plot_correlation(self, timeframe: str, figsize=(12, 10)):
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        corr_matrix = self.calculate_correlation(timeframe)
        if not corr_matrix.empty:
            plt.figure(figsize=figsize)
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title(f'Correlation Matrix - {timeframe}')
            plt.show()

    def __repr__(self):
        return self.__str__()

# =================================================================================================================================|| TRADE

class Trade(BaseClass):
    def __init__(self, params: dict): 
        super().__init__()
        self.asset = params.get('asset')
        self.direction = params.get('direction', 'long')
        self.entry_price = params.get('entry_price', 0.0)
        self.entry_time = params.get('entry_time', datetime.datetime.now())
        self.lot_size = params.get('lot_size', 1.0)
        self.stop_loss = params.get('stop_loss')
        self.take_profit = params.get('take_profit')
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None
        self.profit = None
        self.profit_r = None

    def close(self, close_params: dict):
        self.exit_price = close_params.get('exit_price', 0.0)
        self.exit_time = close_params.get('exit_time', datetime.datetime.now())
        self.exit_reason = close_params.get('exit_reason', 'unknown')
        
        price_diff = (self.exit_price - self.entry_price) if self.direction == 'long' else (self.entry_price - self.exit_price)
        self.profit = price_diff * self.lot_size
        
        if self.stop_loss:
            risk = abs(self.entry_price - self.stop_loss)
            self.profit_r = price_diff / risk if risk != 0 else 0 

# =================================================================================================================================|| MMA

class Money_Management_Algorithm(BaseClass):
    def __init__(self, strat_params: dict):
        super().__init__()
        self.name = strat_params.get('name', 'unnamed_mm')
        
        # Parâmetros existentes...
        self.position_sizing_type = strat_params.get('position_sizing_type', 'percentage')
        self.position_sizing_from = strat_params.get('position_sizing_from', 'balance')
        self.position_sizing_method = strat_params.get('position_sizing_method', 'regular')
        
        self.init_capital = strat_params.get('init_capital', 10000.0)
        self.max_capital_exposure = strat_params.get('max_capital_exposure', 0.75)
        self.max_drawdown = strat_params.get('max_drawdown', 0.5)
        
        self.trade_risk_default = strat_params.get('trade_risk_default', 0.005)
        self.trade_risk_min = strat_params.get('trade_risk_min', 0.001)
        self.trade_risk_max = strat_params.get('trade_risk_max', 0.05)
        self.trade_max_num_open = strat_params.get('trade_max_num_open', 1)
        self.trade_min_num_analysis = strat_params.get('trade_min_num_analysis', 100)
        
        self.confidence_level = strat_params.get('confidence_level', 0.5)
        self.kelly_weight = strat_params.get('kelly_weight', 0.1)
        
        # Para expansão futura
        self.external_data = {}  # Para dados externos como COT
        self.allocation_rules = {}  # Para regras de alocação customizadas

    @staticmethod
    def calculate_risk_perc(trades: list[Trade], type_management: str, risk_default: float=0.005, risk_min: float=0.001, risk_max: float=0.05,
                            min_quant_trades=100, confidence_level=0.5, kelly_weight=0.1):
        
        if len(trades) < min_quant_trades:
            return risk_default
            
        if type_management == 'regular':
            return risk_default
            
        elif type_management == 'kelly':
            wins = [trade for trade in trades if trade.profit > 0]
            losses = [trade for trade in trades if trade.profit <= 0]
            
            if not wins or not losses:
                return risk_default
                
            win_prob = len(wins) / len(trades)
            avg_win = sum(trade.profit_r for trade in wins) / len(wins)
            avg_loss = abs(sum(trade.profit_r for trade in losses) / len(losses))
            
            kelly = win_prob - ((1 - win_prob) / (avg_win / avg_loss))
            kelly *= kelly_weight
            
            return max(min(kelly, risk_max), risk_min)
            
        elif type_management == 'confidence':
            sorted_returns = sorted(trade.profit_r for trade in trades)
            index = int(len(sorted_returns) * confidence_level)
            conf_return = sorted_returns[index]
            
            if conf_return <= 0:
                return risk_min
            return max(min(conf_return * risk_default, risk_max), risk_min)
            
        else:
            return risk_default

    @staticmethod
    def calculate_lot_size(asset_metrics: dict, stop_diff: float, balance: float, risk_percent: float):
        fin_risk = balance * risk_percent
        price_risk = stop_diff * asset_metrics['tick_fin_val']
        
        if price_risk == 0:
            return 0
            
        lot_size = fin_risk / price_risk
        lot_size = max(round(lot_size / asset_metrics['min_lot']) * asset_metrics['min_lot'], asset_metrics['min_lot'])
        
        max_leverage = balance * asset_metrics['leverage']
        max_lot = max_leverage / asset_metrics['lot_value']
        
        return min(lot_size, max_lot)

@dataclass
class MoneyManagementParams:
    """Parâmetros para configurar o Money Management"""
    name: str
    
    # Position Sizing
    position_sizing_type: str = 'percentage'  # 'percentage', 'kelly', 'confidence'
    position_sizing_from: str = 'balance'     # 'balance', 'equity'
    position_sizing_method: str = 'regular'   # 'regular', 'dynamic'
    
    # Capital Management
    init_capital: float = 10000.0
    max_capital_exposure: float = 0.75
    max_drawdown: float = 0.5
    
    # Risk Management
    trade_risk_default: float = 0.005
    trade_risk_min: float = 0.001
    trade_risk_max: float = 0.05
    trade_max_num_open: int = 1
    trade_min_num_analysis: int = 100
    
    # Advanced Parameters
    confidence_level: float = 0.5
    kelly_weight: float = 0.1
    
    # Portfolio Management (para futuro)
    external_data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    allocation_rules: Dict[str, callable] = field(default_factory=dict)














@dataclass
class TimeRules:
    execution_timeframe: str = 'M15'
    day_trade: bool = False
    timeTI: Optional[list[int]] = None
    timeEF: Optional[list[int]] = None
    timeTF: Union[bool, List[int]] = False
    next_index_day_close: bool = False
    timeExcludeHours: Optional[List[int]] = None
    friday_close: bool = False
    dateExcludeTradingDays: Optional[List[int]] = None
    dateExcludeMonths: Optional[List[int]] = None

@dataclass
class TradeManagementRules:
    LONG: bool = True
    SHORT: bool = True
    TF: bool = True
    SL: bool = True
    TP: bool = False
    BE_pos: bool = False
    BE_neg: bool = False
    NB: int = -1

@dataclass
class RiskManagementRules:
    position_sizing_type: str='percentage'
    position_sizing_from: str='balance'
    position_sizing_method: str='regular'

    init_capital: float=10000.0
    max_capital_exposure: float=0.75
    max_drawdown: float=0.5

    trade_risk_default: float=0.005
    trade_risk_min: float=0.001
    trade_risk_max: float=0.05
    trade_max_num_open: int=1
    trade_min_num_analysis: int=100

    confidence_level: float=0.5
    kelly_weight: float=0.1

@dataclass
class ExecutionRules:
    order_type: str='market'
    offset: float=0.0

@dataclass
class DataSettings:
    dataframe: pd.DataFrame = None
    additional_timeframes: List[str] = field(default_factory=lambda: ['D1'])
    fill_method: str='ffill'
    fillna = 0

@dataclass
class Strat_Parameters():
    name: str
    assets: Asset_Portfolio

    execution: ExecutionRules = field(default_factory=ExecutionRules)
    time: TimeRules = field(default_factory=TimeRules)
    trade: TradeManagementRules = field(default_factory=TradeManagementRules)
    risk: RiskManagementRules = field(default_factory=RiskManagementRules)
    data: DataSettings = field(default_factory=DataSettings)

    entry_rules: Dict[str, Callable[[pd.DataFrame], pd.Series]] = field(default_factory=dict)
    tf_exit_rules: Dict[str, Callable[[pd.DataFrame], pd.Series]] = field(default_factory=dict)
    sl_exit_rules: Dict[str, Callable[[pd.DataFrame], pd.Series]] = field(default_factory=dict) 

class Strat(BaseClass):
    def __init__(self, strat_params: Strat_Parameters):
        super().__init__()
        self.name = strat_params.name
        self.assets = strat_params.assets
        self.strat_parameters = strat_params

        # Regras de execução
        self.execution_rules = strat_params.execution
        
        # Regras de tempo
        self.time_rules = strat_params.time
        
        # Regras de trade
        self.trade_rules = strat_params.trade
        
        # Regras de risco
        self.risk_rules = strat_params.risk
        
        # Configurações de dados
        self.data_settings = strat_params.data
        
        # Regras dinâmicas
        self.entry_rules = strat_params.entry_rules
        self.tf_exit_rules = strat_params.tf_exit_rules
        self.sl_exit_rules = strat_params.sl_exit_rules

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
        
        # Aplica regras de direção
        if not self.trade_rules.LONG: df['entry_long'] = False
        if not self.trade_rules.SHORT: df['entry_short'] = False
        
        return df

    def _apply_exit_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        # Time Frame Exit
        if self.trade_rules.TF and self.tf_exit_rules:
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
                df['exit_tf'] = tf_signals.all(axis=1)
        
        # Stop Loss
        if self.trade_rules.SL and self.sl_exit_rules:
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

    def generate_signals(self, df: pd.DataFrame):
        # Aplicar as regras de entrada
        df = self._apply_entry_rules(df)
        
        # Aplicar as regras de saída
        df = self._apply_exit_rules(df)
        
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

class Strat_Portfolio(BaseClass):
    def __init__(self, strat_portfolio_params: dict):
        super().__init__()
        self.name = strat_portfolio_params.name
        self.strats = strat_portfolio_params.strats
        

def monte_carlo(trades: list[Trade], num_simulations=1000, shuffle=True):
    """
    Realiza simulação de Monte Carlo em uma lista de trades
    
    Args:
        trades (list[Trade]): Lista de trades para simular
        num_simulations (int): Número de simulações
        shuffle (bool): Se True, embaralha a ordem dos trades
        
    Returns:
        dict: Dicionário com resultados da simulação
    """
    if not trades:
        return None
        
    results = []
    trades_r = [trade.profit_r for trade in trades if trade.profit_r is not None]
    
    if not trades_r:
        return None
        
    for _ in range(num_simulations):
        if shuffle:
            np.random.shuffle(trades_r)
        equity_curve = np.cumsum(trades_r)
        results.append({
            'final_return': equity_curve[-1],
            'max_dd': min(0, np.min(equity_curve - np.maximum.accumulate(equity_curve))),
            'equity_curve': equity_curve
        })
        
    final_returns = [r['final_return'] for r in results]
    max_dds = [r['max_dd'] for r in results]
    
    return {
        'mean_return': np.mean(final_returns),
        'std_return': np.std(final_returns),
        'mean_dd': np.mean(max_dds),
        'std_dd': np.std(max_dds),
        'worst_dd': min(max_dds),
        'best_return': max(final_returns),
        'worst_return': min(final_returns),
        'equity_curves': [r['equity_curve'] for r in results]
    } 













# def Walkforward(wf_dataframe, wf_optimizations, wf_matrix):

#     # 1. Preparação dos dados
#     wf_data = wf_dataframe.copy()
#     wf_data['datetime'] = pd.to_datetime(wf_data['datetime'])
    
#     # Calcular data mínima e total de meses
#     min_date = wf_data['datetime'].min()
#     max_date = wf_data['datetime'].max()
#     total_months = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month) + 1
    
#     # 2. Processar combinações walkforward
#     wfm_combinations = [(is_size, os_size) for is_size, os_size in itertools.product(*wf_matrix) if is_size >= os_size]
#     results_all = []
#     performance = {}
    
#     for is_months, os_months in wfm_combinations:
#         if (is_months + os_months) > total_months:
#             print(f'!!! --- Warning WF {is_months}IS/{os_months}OS window ignored, not enought data --- !!!')
#             continue
#         print(f"\nWalk Forward Optimization >>> {math.floor((total_months - is_months) / os_months)} Windows out of {total_months} Total Months >>> Matrix: ({is_months}, {os_months})") 
            
#         current_month = 0
#         best_params = None
#         all_os_trades = []
        
#         while current_month + is_months + os_months <= total_months:
#             # Definir janelas temporais
#             is_start = current_month
#             is_end = current_month + is_months
#             os_start = is_end
#             os_end = os_start + os_months
            
#             # 3. Fase IS - Seleção de parâmetros
#             best_perf = -float('inf')
#             current_best = best_params
            
#             for backtest in wf_optimizations:
#                 if not backtest:
#                     continue
                
#                 # Calcular performance IS
#                 perf = 0
#                 for trade in backtest:
#                     trade_date = pd.to_datetime(trade['date_entry'])
#                     trade_month = (trade_date.year - min_date.year) * 12 + (trade_date.month - min_date.month)
                    
#                     if is_start <= trade_month < is_end:
#                         perf += trade['pnl']
                
#                 if perf > best_perf:
#                     best_perf = perf
#                     current_best = backtest[0]['param_combination']

#             # Fallback: usar último best_params ou primeiro disponível
#             if current_best is None: current_best = best_params if best_params is not None else wf_optimizations[0][0]['param_combination'] 
#             else: best_params = current_best

#             # 4. Fase OS - Coletar trades
#             for backtest in wf_optimizations:
#                 if backtest and backtest[0]['param_combination'] == current_best:
#                     perf_os = 0
#                     for trade in backtest:
#                         trade_date = pd.to_datetime(trade['date_entry'])
#                         trade_month = (trade_date.year - min_date.year) * 12 + (trade_date.month - min_date.month)
                        
#                         if os_start <= trade_month < os_end:
#                             trade_copy = trade.copy()
#                             #trade_copy.update({'wf_window': f'{is_months}IS/{os_months}OS', 'wf_params': current_best, 'wf_phase': 'OS'})
#                             all_os_trades.append(trade_copy)
#                             perf_os += trade_copy['pnl']
#                     performance[f"{is_months}/{os_months}"] = {'IS': best_perf, 'OS': perf_os, 'params': current_best}
#                     break
#             current_month += os_months  # Avançar janela
#             print(f"WF Window IS: {is_start+1}-{is_end} | OS: {os_start+1}-{os_end} -> Best IS params: {current_best}")

#         # 5. Consolidar resultados
#         results_all.append(all_os_trades) 
    
#     # --- Análise de Performance ---
#     if performance:
#         print("\n" + "="*50)
#         print("Resultados por Configuração Walkforward (IS vs OS)")
#         print("="*50)
#         print("\n"+"-"*60)
        
#         # Organiza os resultados por configuração de walkforward
#         wf_results = {}
#         for key, values in performance.items():
#             wf_config = key.split('_')[0]  # Extrai a configuração (ex: '(12,4)')
#             if wf_config not in wf_results:
#                 wf_results[wf_config] = []
#             wf_results[wf_config].append(values)
        
#         # Imprime resultados por configuração walkforward
#         for config, results in wf_results.items():
#             is_avg = np.mean([r['IS'] for r in results])
#             os_avg = np.mean([r['OS'] for r in results])
#             var_avg = ((abs(os_avg) - abs(is_avg)) / abs(is_avg)) * 100 if is_avg != 0 else 0
#             if var_avg!= 0 and os_avg < is_avg and var_avg > 0: var_avg = var_avg * -1
#             elif var_avg!= 0 and os_avg > is_avg and var_avg < 0: var_avg = var_avg * -1
            
#             print(f"Parametro WF{config}:")
#             print(f"  Média IS: {is_avg:.4f}")
#             print(f"  Média OS: {os_avg:.4f}")
#             print(f"  Variação Média: {var_avg:.2f}%")
#             print("-"*60)
        
#         # Calcula estatísticas globais
#         all_is = [r['IS'] for config in wf_results.values() for r in config]
#         all_os = [r['OS'] for config in wf_results.values() for r in config]
#         global_is = np.mean(all_is)
#         global_os = np.mean(all_os)
#         global_var = ((abs(global_os) - abs(global_is)) / abs(global_is)) * 100 if global_is != 0 else 0
#         if global_var!= 0 and global_os < global_is and global_var > 0: global_var = global_var * -1 # ARRUMAR BAGULHO
#         elif global_var!= 0 and global_os > global_is and global_var < 0: global_var = global_var * -1
        
#         # Imprime resumo geral
#         print("\n" + "="*60)
#         print("RESUMO GERAL DE TODAS AS CONFIGURAÇÕES:")
#         print(f"  Média Global IS: {global_is:.4f}")
#         print(f"  Média Global OS: {global_os:.4f}")
#         print(f"  Variação Média Global: {global_var:.2f}%")
#         print("="*60 + "\n")

#     return results_all  

# def Backtest(datas, params, backtestConfig, wfm=None, print_trades=False): 
#     param_combinations = list(itertools.product(*params))
#     datas = datas.copy()
#     results_all = []

#     for dataframe in datas: # Acessa cada ativo
#         results_ativo_parametros = []
#         for k, combination in enumerate(param_combinations):
#             df = dataframe[k]
#             ativo = df['ativo'].iloc[-1]

#             ac_account_balance = backtestConfig['initial_capital']
#             ac_max_position_size = backtestConfig['position_size_per_trade']
#             ac_risk_per_trade = backtestConfig['avg_risk_per_trade']
#             ac_max_capital_drawdown = backtestConfig['initial_capital'] * 0.5
#             position_sizing_type = backtestConfig['position_sizing_type']
#             asset_metrics = get_instrument_values(ativo)

#             open = df['open'].copy()
#             high = df['high'].copy()
#             low = df['low'].copy()
#             close = df['close'].copy()
#             datetime_current = df['datetime'].copy()
#             entry_price_long = df['entry_price_long'].copy()
#             entry_price_short = df['entry_price_short'].copy()
#             signals = df[f'entry_{combination}'].copy()
#             exits = df[f'tf_{combination}'].copy()
     
#             if backtestConfig['SL']:
#                 slL = df[f'slL_{combination}'].copy()
#                 slS = df[f'slS_{combination}'].copy()
#             if backtestConfig['TP']:
#                 tpL = df[f'tpL_{combination}'].copy()
#                 tpS = df[f'tpS_{combination}'].copy()
#             if backtestConfig['BE+']: breakeven_pos = df[f'breakeven_pos'].copy()
#             if backtestConfig['BE-']: breakeven_neg = df[f'breakeven_neg'].copy()
#             if backtestConfig['day_trade']: exit_end_of_day = df['exit_end_of_day'].copy()

#             backtest_trades = []
#             trade = None
#             n_bars_exit = 0

#             for i in range(1, len(df)):
#                 datt = datetime_current[i]
#                 entry_confirmed = False
#                 signal = signals[i-1]

#                 # Entry
#                 if trade is None:
#                     if backtestConfig['type_order'] == 'market' and signal != 0: entry_confirmed = True 
#                     elif backtestConfig['type_order'] == 'limit': # Precisaria recalcular caso abertura acima/abaixo desse entrada em limit
#                         #if signal > 0 and open[i] < entry_price_long[i] and high[i] > entry_price_long[i]: entry_confirmed = True
#                         #elif signal < 0 and open[i] > entry_price_short[i] and low[i] < entry_price_short[i]: entry_confirmed = True
#                         if signal > 0:
#                             if high[i] > entry_price_long[i] and open[i] < entry_price_long[i]: 
#                                 entry_confirmed = True
#                             elif open[i] > entry_price_long[i]: 
#                                 entry_confirmed = True
#                                 print(f'!!! --- open[i]: {open[i]} > entry_price_long[i]: {entry_price_long[i]}, disregarding limit order --- !!!')
#                         elif signal < 0:
#                             if low[i] < entry_price_short[i] and open[i] > entry_price_short[i]: 
#                                 entry_confirmed = True
#                             elif open[i] < entry_price_short[i]:
#                                 entry_confirmed = True
#                                 print(f'!!! --- open[i]: {open[i]} < entry_price_short[i]: {entry_price_short[i]}, disregarding limit order --- !!!')

#                     if entry_confirmed:
#                         if backtestConfig['SL']: sl_val = slL[i] if signal > 0 else slS[i]
#                         else: sl_val = None
#                         if backtestConfig['TP']: tp_val = tpL[i] if signal > 0 else tpS[i]
#                         else: tp_val = None
                        
#                         if position_sizing_type not in ['fixed', 'perc_cumulative']: # Use Kelly's Criterion
#                             pnl_diff_values = [trade['pnl_diff'] for trade in backtest_trades if trade.get('pnl_diff') is not None]
#                             ac_account_calc = calculate_risk_perc(pnl_diff_values, position_sizing_type)
#                         else: ac_account_calc = ac_risk_per_trade

#                         if position_sizing_type == 'fixed':
#                             lot_size = calculate_lot_size(ativo, signal, backtestConfig['initial_capital'], ac_account_calc)
#                         else: lot_size = calculate_lot_size(ativo, signal, ac_account_balance, ac_account_calc)
#                         trade = {'date_entry': datt, 
#                                 'date_exit': None, 
#                                 'position_size': lot_size * np.sign(signal), #signal, #math.floor((risk_per_trade * account_balance) / signal),
#                                 'sl': sl_val,
#                                 'tp': tp_val,
#                                 'price_entry': entry_price_long[i] if signal > 0 else entry_price_short[i], 
#                                 'price_exit': None,
#                                 'pnl': None, 
#                                 'pnl_diff': None,
#                                 'bars_in_trade': 0,
#                                 'param_combination': str(combination), 
#                                 'ativo': ativo}
                        
#                         n_bars_exit = 0 #if backtestConfig['type_order'] == 'market' else 1
#                         if print_trades:
#                             side = '+' if signal > 0 else ''
#                             print(f"{i} >>> Balance ${round(ac_account_balance,2)} >>> Entry    at {datt} >>> {side}{np.sign(trade['position_size'])} at {trade['price_entry']} >>> SL: {sl_val} > TP: {tp_val} <---> High: {round(high[i], 2)} Low: {round(low[i], 2)}")

#                 # Exit & Bar to Bar Variation
#                 if trade is not None :
#                     if backtestConfig['day_trade'] and (df['date'].iloc[i] < df['date'].iloc[i-1]): print('!!! --- Warning, DT went overnigth --- !!!')
#                     n_bars_exit += 1
#                     exit = exits[i-1]

#                     sl, tp, end_of_day_exit, trend_following_exit = False, False, False, False
#                     if n_bars_exit == 1 and backtestConfig['type_order'] == 'limit': # 290425 Ultima mod, precisa só pra limit correto?
#                         if trade['sl'] is not None:                      
#                             if trade['position_size'] > 0: sl = ((close[i] < open[i]) and (low[i] <= trade['sl']))
#                             elif trade['position_size'] < 0: sl = ((close[i] > open[i]) and (high[i] >= trade['sl']))
#                         if trade['tp'] is not None:
#                             if trade['position_size'] > 0: tp = ((close[i] > open[i]) and (high[i] > trade['tp']))
#                             elif trade['position_size'] < 0: tp = ((close[i] < open[i]) and (low[i] < trade['tp']))
#                     else:
#                         if trade['sl'] is not None:                      
#                             if trade['position_size'] > 0: sl = (low[i] <= trade['sl']) 
#                             elif trade['position_size'] < 0: sl = (high[i] >= trade['sl']) 
#                         if trade['tp'] is not None:
#                             if trade['position_size'] > 0: tp = (high[i] > trade['tp']) 
#                             elif trade['position_size'] < 0: tp = (low[i] < trade['tp']) 

#                     if sl and tp: # Entrada limite, caso SL tenha sido atingido verifica ordem barra para definir se SL ou Entrada
#                         print(f'!!! --- SL and TP triggered at the same time, checking candle direction to define if SL or TP --- !!!')
#                         if trade['position_size'] > 0:
#                             if close[i] > open[i]: tp = False
#                             else: sl = False
#                         elif trade['position_size'] < 0:
#                             if close[i] < open[i]: tp = False
#                             else: sl = False

#                     if backtestConfig['day_trade'] and exit_end_of_day[i]: end_of_day_exit = True 
#                     if backtestConfig['TF'] and exit == np.sign(trade['position_size']) and n_bars_exit > 1: trend_following_exit = True 

#                     if trend_following_exit or sl or tp or (n_bars_exit > backtestConfig['NB']) or end_of_day_exit or (i >= len(df)-1) : #
#                         side = '+' if trade['position_size'] > 0 else ''

#                         if trend_following_exit or n_bars_exit > backtestConfig['NB'] or end_of_day_exit or (i >= len(df)-1): trade['price_exit'] = open[i] 
#                         elif sl: trade['price_exit'] = trade['sl']
#                         elif tp: trade['price_exit'] = trade['tp']

#                         trade['date_exit'] = datt
#                         trade['bars_in_trade'] = n_bars_exit

#                         # Método 1 - Parece ser o correto
#                         trade['pnl_diff'] = ((trade['price_exit'] - trade['price_entry']) / asset_metrics['tick'] * asset_metrics['tick_fin_val'] * trade['position_size']) - (
#                             ((asset_metrics['comissions'] + asset_metrics['slippage'] + asset_metrics['spread']) * backtestConfig['comission_mult'] * abs(trade['position_size'])))
                        
#                         trade['pnl'] = trade['pnl_diff'] / backtestConfig['initial_capital']

#                         #trade['pnl'] = (((trade['price_exit'] - trade['price_entry'])) / trade['price_entry']) * trade['position_size']
#                         ac_account_balance += trade['pnl_diff']

#                         if print_trades:
#                             if trend_following_exit: print(f"{i} >>> Balance ${round(ac_account_balance,2)} >>> TF Exit{' '} at {datt} >>> {side}{np.sign(trade['position_size'])} at {round(trade['price_exit'])} >>> {round(trade['pnl'],2) * 100}% = {round(trade['pnl_diff'], 2)} <---> High: {round(high[i], 2)} Low: {round(low[i], 2)}")
#                             if (n_bars_exit > backtestConfig['NB']): print(f"{i} >>> Balance ${round(ac_account_balance,2)} >>> NB Exit{' '} at {datt} >>> {side}{np.sign(trade['position_size'])} at {round(trade['price_exit'])} >>> {round(trade['pnl'] * 100, 2)}% = {round(trade['pnl_diff'], 2)} <---> High: {round(high[i], 2)} Low: {round(low[i], 2)}")           
#                             if (i >= len(df)-1): print(f"{i} >>> Balance ${round(ac_account_balance,2)} >>> EF Exit{' '} at {datt} >>> {side}{np.sign(trade['position_size'])} at {round(trade['price_exit'])} >>> {round(trade['pnl'] * 100, 2)}% = {round(trade['pnl_diff'], 2)} <---> High: {round(high[i], 2)} Low: {round(low[i], 2)}")                                             
#                             if end_of_day_exit: print(f"{i} >>> Balance ${round(ac_account_balance,2)} >>> ED Exit{' '} at {datt} >>> {side}{np.sign(trade['position_size'])} at {round(trade['price_exit'])} >>> {round(trade['pnl'] * 100, 2)}% = {round(trade['pnl_diff'], 2)} <---> High: {round(high[i], 2)} Low: {round(low[i], 2)}")
#                             if sl: print(f"{i} >>> Balance ${round(ac_account_balance,2)} >>> SL Exit{' '} at {datt} >>> {side}{np.sign(trade['position_size'])} at {round(trade['price_exit'])} >>> {round(trade['pnl'] * 100, 2)}% = {round(trade['pnl_diff'], 2)} <---> High: {round(high[i], 2)} Low: {round(low[i], 2)}")
#                             if tp: print(f"{i} >>> Balance ${round(ac_account_balance,2)} >>> TP Exit{' '} at {datt} >>> {side}{np.sign(trade['position_size'])} at {round(trade['price_exit'])} >>> {round(trade['pnl'] * 100, 2)}% = {round(trade['pnl_diff'], 2)} <---> High: {round(high[i], 2)} Low: {round(low[i], 2)}")

#                         n_bars_exit = 0
#                         backtest_trades.append(trade)
#                         trade = None 
#                         sl = False
#                         tp = False

#                     elif backtestConfig['BE+'] and trade['sl'] is not None: #Tratar Break Even e (Acompanhamento de cada posição?)
#                         if trade['position_size'] > 0:
#                             be_level = trade['price_entry'] + breakeven_pos[i] if breakeven_pos[i] != -1 else trade['price_entry'] + (trade['price_entry'] - trade['sl'])
#                             if high[i] >= be_level and trade['sl'] != trade['price_entry']:
#                                 trade['sl'] = trade['price_entry']
#                                 if print_trades: print(f"{i} >>> BE+      at {datt} >>> {side}{np.sign(trade['position_size'])} at {be_level} >>> SL: {trade['sl']} = EN: {trade['price_entry']} <---> High: {round(high[i], 2)} Low: {round(low[i], 2)}")
#                         elif trade['position_size'] < 0:
#                             be_level = trade['price_entry'] - breakeven_pos[i] if breakeven_pos[i] != -1 else trade['price_entry'] - (trade['sl'] - trade['price_entry'])
#                             if low[i] <= be_level and trade['sl'] != trade['price_entry']:
#                                 trade['sl'] = trade['price_entry']
#                                 if print_trades: print(f"{i} >>> BE+      at {datt} >>> {side}{np.sign(trade['position_size'])} at {be_level} >>> SL: {trade['sl']} = EN: {trade['price_entry']} <---> High: {round(high[i], 2)} Low: {round(low[i], 2)}")
                                
#                     elif backtestConfig['BE-'] and trade['tp'] is not None: #Tratar Break Even e (Acompanhamento de cada posição?)
#                         if trade['position_size'] > 0:
#                             be_level = trade['price_entry'] - breakeven_neg[i] if breakeven_neg[i] != -1 else trade['price_entry'] - (trade['tp'] - trade['price_entry'])
#                             if low[i] <= be_level and trade['tp'] != trade['price_entry']:
#                                 trade['tp'] = trade['price_entry']
#                                 if print_trades: print(f"{i} >>> BE-      at {datt} >>> {side}{np.sign(trade['position_size'])} at {be_level} >>> SL: {trade['tp']} = EN: {trade['price_entry']} <---> High: {round(high[i], 2)} Low: {round(low[i], 2)}")
#                         elif trade['position_size'] < 0:
#                             be_level = trade['price_entry'] + breakeven_neg[i] if breakeven_neg[i] != -1 else trade['price_entry'] + (trade['price_entry'] - trade['tp'])
#                             if high[i] >= be_level and trade['tp'] != trade['price_entry']:
#                                 trade['tp'] = trade['price_entry']
#                                 if print_trades: print(f"{i} >>> BE-      at {datt} >>> {side}{np.sign(trade['position_size'])} at {be_level} >>> SL: {trade['tp']} = EN: {trade['price_entry']} <---> High: {round(high[i], 2)} Low: {round(low[i], 2)}")

#             # Verifica se todos os returns são zero ou NaN
#             if not backtest_trades or all(trade['pnl'] is None or trade['pnl'] == 0 for trade in backtest_trades):
#                 print(f"!!! --- Warning: All returns are zero or NaN for combination {combination} --- !!!")
#                 continue

#             results_ativo_parametros.append(backtest_trades)

#         if wfm is not None: 
#             #results_ativo_parametros = Concat_Backtest_Results(results_ativo_parametros)
#             results_ativo_parametros = Walkforward(dataframe[-1], results_ativo_parametros, wfm)
#             results_all.append(results_ativo_parametros)
#         else: results_all.append(results_ativo_parametros)
#     return results_all


# def print_backtest_metrics(all_long_strat_pnls, all_long_strat_pnls_diff, all_short_strat_pnls, all_short_strat_pnls_diff,
#                            return_financial, return_financial_trade, return_perc, return_perc_trade, profit_factor, payoff, exp_r, exposition,
#                            drawdown_50, drawdown_95, stagnation_50, stagnation_95, runs_test_result, trade_dependency_result, ret_without_top_perc,
#                            ret_top_perc, win_perc, loss_perc, even_perc, len_backtests, trades, max_ret, benchmark_pnl_perc):
    
#     # Long Trades
#     trades_long = [pnl for pnl_list in all_long_strat_pnls_diff for pnl in pnl_list]
#     trades_perc_long = [pnl for pnl_list in all_long_strat_pnls for pnl in pnl_list]
#     wins_long = [pnl for pnl in trades_long if pnl > 0]
#     losses_long = [pnl for pnl in trades_long if pnl < 0]
#     break_even_long = [pnl for pnl in trades_long if pnl == 0]

#     if len(trades_long) > 0:
#         win_perc_long = (len(wins_long)/((len(wins_long)+len(losses_long)+len(break_even_long))))*100
#         loss_perc_long = (len(losses_long)/(len(wins_long)+len(losses_long)+len(break_even_long)))*100
#         even_perc_long = 100 - (win_perc_long + loss_perc_long)
#         try: payoff_long = (sum(wins_long)/len(wins_long))/((sum(losses_long)*-1)/len(losses_long)) 
#         except: payoff_long = 0
#         return_financial_long = np.sum(trades_long)/len_backtests
#         return_financial_trade_long = np.sum(return_financial_long)/(len(trades_long)/len_backtests)
#         return_perc_long = ((np.sum(trades_perc_long))/len_backtests)*100
#         return_perc_trade_long = ((np.sum(return_perc_long))/(len(trades_long)/len_backtests))*100
#         profit_factor_long = sum(wins_long)/(sum(losses_long)*-1)
#         exp_r_long = (win_perc_long * payoff_long) - loss_perc_long
#     else: 
#         win_perc_long=0
#         loss_perc_long=0
#         even_perc_long=0
#         payoff_long=0
#         return_financial_long=0
#         return_financial_trade_long=0
#         return_perc_long=0
#         return_perc_trade_long=0
#         profit_factor_long=0
#         exp_r_long=0

#     # Short Trades
#     trades_short = [pnl for pnl_list in all_short_strat_pnls_diff for pnl in pnl_list]
#     trades_perc_short = [pnl for pnl_list in all_short_strat_pnls for pnl in pnl_list]
#     wins_short = [pnl for pnl in trades_short if pnl > 0]
#     losses_short = [pnl for pnl in trades_short if pnl < 0]

#     if len(trades_short) > 0:
#         win_perc_short = (len(wins_short)/(len(wins_short)+len(losses_short)+len(break_even_long)))*100
#         loss_perc_short = (len(losses_short)/(len(wins_short)+len(losses_short)+len(break_even_long)))*100
#         even_perc_short = 100 - (win_perc_short + loss_perc_short)
#         try: payoff_short = (sum(wins_short)/len(wins_short))/((sum(losses_short)*-1)/len(losses_short)) 
#         except: payoff_short = 0
#         return_financial_short = np.sum(trades_short)/len_backtests
#         return_financial_trade_short = np.sum(return_financial_short)/(len(trades_short)/len_backtests)
#         return_perc_short = ((np.sum(trades_perc_short))/len_backtests)*100
#         return_perc_trade_short = ((np.sum(return_perc_short))/(len(trades_short)/len_backtests))*100
#         profit_factor_short = sum(wins_short)/(sum(losses_short)*-1)
#         exp_r_short = (win_perc_short * payoff_short) - loss_perc_short
#     else: 
#         win_perc_short=0
#         loss_perc_short=0
#         even_perc_short=0
#         payoff_short=0
#         return_financial_short=0
#         return_financial_trade_short=0
#         return_perc_short=0
#         return_perc_trade_short=0
#         profit_factor_short=0
#         exp_r_short=0

#     if len(trades) > 0:
#         max_all = (return_perc/max_ret[0])*100
#         max_pos = (return_perc_long/max_ret[1])*100
#         max_neg = (return_perc_short/max_ret[2])*100

#         ret_exp = (return_perc/(exposition/100))
#         ret_exp_pos = (return_perc_long/(exposition/100))
#         ret_exp_neg = (return_perc_short/(exposition/100))
  
#         alpha = return_perc - benchmark_pnl_perc
#         alpha_long = alpha * (return_perc_long / return_perc)
#         alpha_short = alpha * (return_perc_short / return_perc)


#     print("\n"+"||"+"="*70+"||"+"\n")
#     print(f"{'METRICS':<40}{'ALL':>10}{'LONG':>10}{'SHORT':>10}")
#     print(f"{'Return ($):':<40}{return_financial:>10.0f}{return_financial_long:>10.0f}{return_financial_short:>10.0f}")
#     print(f"{'Return (%):':<40}{return_perc:>10.2f}{return_perc_long:>10.2f}{return_perc_short:>10.2f}")
#     print(f"{'Return per Trade ($):':<40}{return_financial_trade:>10.2f}{return_financial_trade_long:>10.2f}{return_financial_trade_short:>10.2f}")
#     print(f"{'Return per Trade (%):':<40}{return_perc_trade:>10.4f}{return_perc_trade_long:>10.4f}{return_perc_trade_short:>10.4f}")
#     print(f"{'Payoff:':<40}{payoff:>10.2f}{payoff_long:>10.2f}{payoff_short:>10.2f}")
#     print(f"{'Profit Factor:':<40}{profit_factor:>10.2f}{profit_factor_long:>10.2f}{profit_factor_short:>10.2f}")
#     print(f"{'Exp R:':<40}{exp_r:>10.2f}{exp_r_long:>10.2f}{exp_r_short:>10.2f}")
#     print(f"{'Return Alpha (%):':<40}{alpha:>10.2f}{alpha_long:>10.2f}{alpha_short:>10.2f}")
#     print(f"{'Return / Max (%):':<40}{max_all:>10.2f}{max_pos:>10.2f}{max_neg:>10.2f}")
#     print(f"{'Return / Exp (%):':<40}{ret_exp:>10.2f}{ret_exp_pos:>10.2f}{ret_exp_neg:>10.2f}")
#     print()
#     print(f"{'Trades:':<40}{len(trades)/len_backtests:>10.0f}{len(trades_long)/len_backtests:>10.0f}{len(trades_short)/len_backtests:>10.0f}") 
#     print(f"{'Wins:':<40}{win_perc:>10.2f}{win_perc_long:>10.2f}{win_perc_short:>10.2f}")
#     print(f"{'Loss:':<40}{loss_perc:>10.2f}{loss_perc_long:>10.2f}{loss_perc_short:>10.2f}")
#     print(f"{'Even:':<40}{even_perc:>10.2f}{even_perc_long:>10.2f}{even_perc_short:>10.2f}")
#     print()
#     print(f"{'Backtests:':<40}{len_backtests:>10.0f}")
#     print(f"{'Exposition (%):':<40}{exposition:>10.2f}") 
#     if drawdown_50: 
#         print(f"{'DD 50th:':<40}{drawdown_50:>10.2f}") 
#         print(f"{'DD 95th:':<40}{drawdown_95:>10.2f}") 
#         print(f"{'Stagnation 50th:':<40}{stagnation_50:>10.2f}") 
#         print(f"{'Stagnation 95th:':<40}{stagnation_95:>10.2f}") 
#     print()
#     print(f"{'Runs Test:':<40}{(runs_test_result):>10.2f}") 
#     print(f"{'Trade Dependency Test PnL:':<40}{trade_dependency_result:>10.0f}") 
#     print(f"{'Return % without Top 3%:':<40}{ret_without_top_perc:>10.2f}")  
#     print(f"{'Top 3% Best Trades Return %:':<40}{ret_top_perc:>10.2f}")  
#     print("\n"+"||"+"="*70+"||"+"\n")


# def run_strategy(config, params, strategy_module, wfm=None, data_file_name=['WIN$_M10.xlsx'], data_file_path='C:\\Users\\Patrick\\Desktop\\Artaxes Portfolio\\MAIN\\MT5_Dados'): 
#     start_time = time.time()
#     strat_name = strategy_module.__name__
#     data, dataBacktests = [], []

#     # Reads all files from list or path in path to select assets if ativo is not selected
#     data = loadExcelCsvFiles(data_file_path, data_file_name, 0, config['TrainTestVal'], index_col_setting=False, drop_index=True)

#     # =================================================================================================================||

#     backtestConfig = {'day_trade': config['day_trade'], 'type_order': config['type_order'], 'timeTI': config['timeTI'], 'timeEF': config['timeEF'], 'timeTF': config['timeTF'],
#                       'SL': config['SL'], 'TP': config['TP'], 'TF': config['TF'], 'BE+': config['BE+'], 'BE-': config['BE-'], 'NB': config['NB'],
#                       'initial_capital': config['initial_capital'], 'position_size_per_trade': config['position_size_per_trade'], 
#                       'avg_risk_per_trade': config['avg_risk_per_trade'], 'comission_mult': config['comission_mult'], 'position_sizing_type': config['position_sizing_type']
#                       }

#     dataStrats = strategy_module(data, params, backtestConfig, strat_name)
#     if wfm is None: dataBacktests = Backtest(dataStrats, params, backtestConfig, None, config['printTrades'])
#     else: dataBacktests = Backtest(dataStrats, params, backtestConfig, wfm, config['printTrades'])

#     # =================================================================================================================||
    
#     backtests = {}

#     for i, ativos_backtest in enumerate(dataBacktests):
#         if config['permutationTEST']: permutation_data_list = calculates_permutation_data(data[i], config['n_permutations']) # Calcula a permutação dos dados apenas n*1 vez para cada ativo
#         for j, backtest in enumerate(ativos_backtest):
#             backtest_ativo = backtest[0]['ativo'] if backtest else 'na'
#             backtest_params = backtest[0]['param_combination'] if backtest else 'na'
#             if wfm is None: dataframe_strategy_column_name = f"{strat_name}_{backtest_ativo}_{backtest_params}"
#             else: dataframe_strategy_column_name = f"{strat_name}_{backtest_ativo}_WFM"

#             #all_backtests.append(backtest)
#             backtest_date_entries = [trade['date_entry'] for trade in backtest if trade['date_entry'] is not None]
#             backtest_pnls_exits = [trade['date_exit'] for trade in backtest if trade['date_exit'] is not None]
#             backtest_positions = [trade['position_size'] for trade in backtest if trade['position_size'] is not None]
#             #backtest_sls = [trade['sl'] for trade in backtest if trade['sl'] is not None]
#             #backtest_tps = [trade['tp'] for trade in backtest if trade['tp'] is not None]
#             #backtest_price_entries = [trade['price_entry'] for trade in backtest if trade['price_entry'] is not None]
#             #backtest_price_exits = [trade['price_exit'] for trade in backtest if trade['price_exit'] is not None]
#             backtest_pnls = [trade['pnl'] for trade in backtest if trade['pnl'] is not None]
#             backtest_pnls_diff = [trade['pnl_diff'] for trade in backtest if trade['pnl_diff'] is not None]
#             backtest_bars_in_trade = [trade['bars_in_trade'] for trade in backtest if trade['bars_in_trade'] is not None]
#             backtest_param_comb = [trade['param_combination'] for trade in backtest if trade['param_combination'] is not None]

#             pd.DataFrame(backtest_pnls_diff).to_excel(f'C:\\Users\\Patrick\\Desktop\\Artaxes Portfolio\\MAIN\\{strat_name}_backtest_pnls_diff.xlsx')
            
#             backtests[dataframe_strategy_column_name] = {
#                 'all_date_entry': backtest_date_entries,
#                 'all_date_exit': backtest_pnls_exits,
#                 'all_position_size': backtest_positions,
#                 #'all_sl': backtest_sls,
#                 #'all_tp': backtest_tps,
#                 #'all_price_entries': backtest_price_entries,
#                 #'all_price_exits': backtest_price_exits,
#                 'all_pnl': backtest_pnls,
#                 'all_pnl_diff': backtest_pnls_diff,
#                 #'all_bars_in_trade': backtest_bars_in_trade,
#                 'all_params': backtest_param_comb,
#                 'ativo': backtest_ativo,
#                 'exposition': sum(backtest_bars_in_trade) / len(data[i]['close'])
#                 }

#             if isinstance(backtest_params, str):               
#                 params_tuple = ast.literal_eval(backtest_params) # Converte a string de volta para tupla
#                 backtest_params_tuple = [[x] for x in params_tuple] # Converte para o formato [[param1], [param2], [param3]]

#             # -> Saves Backtest
#             if config['saveTestsPlots']: SaveToExcel(backtest, dataframe_strategy_column_name, False, 'C:\\Users\\Patrick\\Desktop\\Artaxes Portfolio\\MAIN\\MT5_Strategies')

#             # -> Teste de Monte Carlo
#             if config['montecarloTEST']:
#                 backtests[dataframe_strategy_column_name]['all_monte_carlo'] = MonteCarlo(backtests[dataframe_strategy_column_name]['all_pnl'], config['n_montecarlo'], None, None, config['mc_shuffle'])
#                 backtests[dataframe_strategy_column_name]['all_monte_carlo_drawdowns'], backtests[dataframe_strategy_column_name]['all_monte_carlo_stagnations'] = calculate_performance_metrics(backtests[dataframe_strategy_column_name]['all_monte_carlo'], 100.0)
                
#             # -> Teste de Permutação
#             if config['permutationTEST']:                                               # NOTE NÃO ESTÀ GERANDO DADOS PERMUTADOS PARA EURUSD_D1 CORRETAMENTE NOTE
#                 permuted_pfs, pval = PT(permutation_data_list, backtests[dataframe_strategy_column_name]['all_pnl'], config, backtest_params_tuple, strategy_module, Backtest, config['n_permutations']) 
#                 permuted_pfs_sum = [np.sum(pf) for pf in permuted_pfs]
#                 backtests[dataframe_strategy_column_name]['p_val'] = pval
#                 plotHist(permuted_pfs_sum, np.sum(backtests[dataframe_strategy_column_name]['all_pnl']), 30, 'blue', config['showTestsPlots'], config['saveTestsPlots'], f"{dataframe_strategy_column_name}_PT_PVal_{pval}", "", "", f"{strat_name} - Permutation Test P-Value: {pval}")
            
#     # =================================================================================================================||

#     # Concatenated Backtest Results % to plot in lines -> Makes them the same len
#     equalized_trade_pnl_for_ploting = Concat_Backtest_Results(backtests)

#     all_strat_pnls, all_strat_pnls_diff, all_strat_positions, all_strat_monte_carlo_drawdowns, all_strat_expositions, all_strat_monte_carlo_stagnations = [], [], [], [], [], []
#     all_long_strat_pnls, all_long_strat_pnls_diff, all_short_strat_pnls, all_short_strat_pnls_diff = [], [], [], []
#     all_strat_pnls_concatenated, all_strat_pnls_diff_concatenated, all_strat_positions_concatenated, all_strat_pnls_concatenated_sum = [], [], [], []
 
#     for strat in equalized_trade_pnl_for_ploting:
#         all_strat_pnls_concatenated.append(equalized_trade_pnl_for_ploting[strat]['all_pnl'])
#         all_strat_pnls_diff_concatenated.append(equalized_trade_pnl_for_ploting[strat]['all_pnl_diff'])
#         all_strat_positions_concatenated.append(equalized_trade_pnl_for_ploting[strat]['all_position_size'])
#         all_strat_pnls_concatenated_sum.append(np.sum(equalized_trade_pnl_for_ploting[strat]['all_pnl']))
        
#     for strat in backtests:
#         # Get all trades data
#         pnls = backtests[strat]['all_pnl']
#         pnl_diffs = backtests[strat]['all_pnl_diff']
#         positions = backtests[strat]['all_position_size']

#         all_strat_pnls.append(pnls)
#         all_strat_pnls_diff.append(pnl_diffs)
#         all_strat_positions.append(positions)
        
#         # Separate long and short trades
#         long_pnls = []
#         long_pnl_diffs = []
#         short_pnls = []
#         short_pnl_diffs = []
        
#         for pnl, pnl_diff, pos in zip(pnls, pnl_diffs, positions):
#             if pos > 0:  # Long trades
#                 long_pnls.append(pnl)
#                 long_pnl_diffs.append(pnl_diff)
#             elif pos < 0:  # Short trades
#                 short_pnls.append(pnl)
#                 short_pnl_diffs.append(pnl_diff)
        
#         # Append to long/short lists
#         all_long_strat_pnls.append(long_pnls)
#         all_long_strat_pnls_diff.append(long_pnl_diffs)
#         all_short_strat_pnls.append(short_pnls)
#         all_short_strat_pnls_diff.append(short_pnl_diffs)

#     for strat in backtests:
#         all_strat_expositions.append(backtests[strat]['exposition'])
#         if config['montecarloTEST']:
#             all_strat_monte_carlo_drawdowns.append(backtests[strat]['all_monte_carlo_drawdowns'])
#             all_strat_monte_carlo_stagnations.append(backtests[strat]['all_monte_carlo_stagnations'])
#     len_backtests = len(backtests)

#     # -> Plot Aggregated PnL by Month
#     if config['showTestsPlots']:
#         agg_pnl_date = aggregate_pnl_by_month(dataBacktests)
#         plotAggReturn(agg_pnl_date, "", "", f"{strat_name} - Aggregated PnL", config['showTestsPlots'], config['saveTestsPlots'], f"{strat_name} - Aggregated PnL")

#         # -> Plot all curves and Distribution of PnL % Results
#         plotLineMult(all_strat_pnls_diff_concatenated, True, config['showTestsPlots'], config['saveTestsPlots'], f"{strat_name} - Equity Curves", "", '', "", False, False)

#     # -> Runs Test
#     pnl_diff_np = np.array(all_strat_pnls_concatenated)
#     mask = ~np.isclose(pnl_diff_np, 0.0, atol=1e-10) # Elimina 0 e -0 (Break Even) 
#     pnl_diff_runs_filtered = pnl_diff_np[mask] 
#     runs_test_result = runs_Test(np.where(np.array(pnl_diff_runs_filtered) > 0, 1, -1))

#     # -> Trade Dependency Test
#     trade_dependency_result_arr = trade_Dependency_Test(np.array(all_strat_pnls_concatenated), 2, 1)

#     # -> Teste In Sample Out Sample
#     if config['showTestsPlots']:
#         if config['in_sample_out_sampleTEST'] and config['TrainTestVal'] > 0.35: # NOTE sharpe_ratio vai dar warning se [0.5, 0.5] validation=0? NOTE
#             if config['TrainTestVal'] <= 0.70: ISOS(all_strat_pnls, [0.5, 0.5], True, False, config['n_montecarlo'], config['mc_shuffle'], config['saveTestsPlots'], f"{strat_name} - IS-OS") 
#             else: ISOS(all_strat_pnls, [0.35, 0.35], True, False, config['n_montecarlo'], config['mc_shuffle'], config['saveTestsPlots'], f"{strat_name} - IS-OS")

#         # -> Distribuição de Variações
#         if config['distributionTEST']:
#             all_individual_returns_percent = np.clip(np.concatenate(all_strat_pnls_diff), -100000, 100000) # Converter para porcentagem e filtrar outliers extremos
#             plotHist(all_individual_returns_percent, None, n_bins=100, colorir='green', show=config['showTestsPlots'], save=config['saveTestsPlots'], saveName=f"{strat_name} - Distribuição de Retornos", xlab="Retorno (%)", ylab="Frequência", title=f"{strat_name} - Distribuição de Retornos", quantiles=True)
#             if len_backtests >= 10: plotHist(all_strat_pnls_concatenated_sum, None, n_bins=len_backtests, colorir='green', show=config['showTestsPlots'], save=config['saveTestsPlots'], saveName=f"{strat_name} - Results", xlab="PnL (%)", ylab="Frequência", title=f"{strat_name} - Distribuição", quantiles=True)

#         # -> Drawdown Máximo Monte Carlo
#         if config['montecarloTEST']: plotHist(all_strat_monte_carlo_drawdowns, None, n_bins=100, colorir='red', show=config['showTestsPlots'], save=config['saveTestsPlots'], saveName=f"{strat_name} - Distribuição de Drawdowns", xlab="Drawdown (%)", ylab="Frequência", title=f"{strat_name} - Distribuição de Drawdowns", quantiles=True)

#     # -> Metrics (Avg over all backtests)
#     metrics_return = {}

#     trades = [pnl for pnl_list in all_strat_pnls_diff for pnl in pnl_list]
#     trades_perc = [pnl for pnl_list in all_strat_pnls for pnl in pnl_list]
#     wins = [pnl for pnl in trades if pnl > 0]
#     losses = [pnl for pnl in trades if pnl < 0]
#     break_even = [pnl for pnl in trades if pnl == 0]

#     positions = [pos for pos_list in all_strat_positions for pos in pos_list]

#     # -> Teste sem os 3% dos maiores trades positivos
#     all_strat_filtered_pnls = [] 
#     wins_perc = [pnl for pnl in trades_perc if pnl > 0]
#     if wins_perc:  # Só calcula se houver trades positivos
#         threshold = np.percentile(wins_perc, 97) #Calcular threshold (97º percentil)
#         percentile_positive = [x for x in wins_perc if x <= threshold] #Filtrar trades abaixo do threshold (excluindo top 3%)
#         percentile_positive_just_best = [x for x in wins_perc if x > threshold]
#         all_strat_filtered_pnls.append(percentile_positive) # Adiciona à lista final
#     else: all_strat_filtered_pnls.append([])  # Caso não haja trades positivos

#     if len(trades) > 0:
#         win_perc = (len(wins)/(len(wins)+len(losses)+len(break_even)))*100
#         loss_perc = (len(losses)/(len(wins)+len(losses)+len(break_even)))*100
#         even_perc = 100-(win_perc+loss_perc)
#         try: payoff = (sum(wins)/len(wins))/((sum(losses)*-1)/len(losses)) 
#         except: payoff = 0
#         return_financial = np.sum(trades)/len_backtests
#         return_financial_trade = np.sum(return_financial)/(len(trades)/len_backtests)
#         return_perc = ((np.sum(trades_perc))/len_backtests)*100
#         return_perc_trade = ((np.sum(return_perc))/(len(trades)/len_backtests))*100
#         profit_factor = sum(wins)/(sum(losses)*-1)
#         exp_r = (win_perc * payoff) - loss_perc
#         exposition = np.percentile(all_strat_expositions, 50)*100
#         # Retorno Máximo do ativo
#         max_ret, max_pos, max_neg = calculate_max_close_open_return(data[i])
#         max_ret = np.sum(max_ret) * 100
#         max_pos = np.sum(max_pos) * 100
#         max_neg = np.sum(max_neg) * 100
#     else:
#         win_perc=0
#         loss_perc=0
#         even_perc=0
#         payoff=0
#         return_financial=0
#         return_financial_trade=0
#         return_perc=0
#         return_perc_trade=0
#         profit_factor=0
#         exp_r=0
#         exposition=0
#         max_ret=0
#         max_pos=0
#         max_neg=0

#     drawdown_50, drawdown_95, stagnation_50, stagnation_95 = 0, 0, 0, 0
#     if config['montecarloTEST']:
#         drawdown_50 = np.percentile(all_strat_monte_carlo_drawdowns, 50) 
#         drawdown_95 = np.percentile(all_strat_monte_carlo_drawdowns, 95)
#         stagnation_50 = np.percentile(all_strat_monte_carlo_stagnations, 50)*100
#         stagnation_95 = np.percentile(all_strat_monte_carlo_stagnations, 95)*100

#     trade_dependency_result = (np.sum(trade_dependency_result_arr))/len_backtests
#     ret_without_top_perc = (np.sum(all_strat_filtered_pnls)*100)/len_backtests
#     ret_top_perc = (np.sum(percentile_positive_just_best)*100)/len_backtests
#     benchmark_pnl_perc = (data[i]['close'].iloc[-1] - data[i]['close'].iloc[0]) / data[i]['close'].iloc[0]

#     print_backtest_metrics(all_long_strat_pnls, all_long_strat_pnls_diff, all_short_strat_pnls, all_short_strat_pnls_diff,
#                             return_financial, return_financial_trade, return_perc, return_perc_trade, profit_factor, payoff, exp_r, exposition,
#                             drawdown_50, drawdown_95, stagnation_50, stagnation_95, runs_test_result, trade_dependency_result, ret_without_top_perc,
#                             ret_top_perc, win_perc, loss_perc, even_perc, len_backtests, trades, [max_ret, max_pos, max_neg], benchmark_pnl_perc)

#     if len(backtests) == 1:
#         metrics_return = {'win_perc': win_perc, 'loss_perc': loss_perc, 'even_perc': even_perc, 'payoff': payoff, 'return_nom': return_financial, 'return_nom_trade': return_financial_trade,
#                             'return_perc': return_perc, 'return_perc_trade': return_perc_trade, 'profit_factor': profit_factor, 'exp_r': exp_r, 'exposition': exposition,
#                             'drawdown_50': drawdown_50, 'drawdown_95': drawdown_95, 'stagnation_50': stagnation_50, 'stagnation_95': stagnation_95, 
#                             'trade_dependency_result': trade_dependency_result, 'ret_without_top_perc': ret_without_top_perc, 'ret_top_perc': ret_top_perc,
#                             'trades': len(trades), 'long': len(all_long_strat_pnls_diff[0]), 'short': len(all_short_strat_pnls_diff[0])
#                             } 
#         backtests[dataframe_strategy_column_name].update(metrics_return)
    
#     # =================================================================================================================||
           
#     end_time = time.time()
#     duration = end_time - start_time
#     print(f'Total backtest/optimization and tests execution time: {round(duration, 2)} seconds\n')
#     return backtests[dataframe_strategy_column_name] #metrics_return, all_pnl_curves, all_pnl_entry_date_entry, all_pnl_entry_date_exit, all_pnl_position_sizes, all_drawdowns, all_individual_returns_percent, permuted_pfs_sum

















