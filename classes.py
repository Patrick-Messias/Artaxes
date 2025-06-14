import pandas as pd
import numpy as np
import os
import sys
import re
import itertools
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Union, Callable, Optional, Any, Type
import sqlite3
import json

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
        self.asset_portfolio = strat_params.assets
        self.execution = strat_params.execution
        self.time_rules = strat_params.time
        self.trade_rules = strat_params.trade
        self.risk_rules = strat_params.risk
        self.data_settings = strat_params.data
        
        # Novo: cache de indicadores
        self.indicators_cache = IndicatorsCache()
        
    def calculate_indicators(self, df: pd.DataFrame, asset_name: str) -> pd.DataFrame:
        """
        Calcula indicadores com suporte a cache
        
        Args:
            df: DataFrame com dados do ativo
            asset_name: Nome do ativo para cache
        """
        df_before = df.copy()
        df_result = self._calculate_indicators(df)
        
        # Salva novos indicadores no cache
        params = self._get_indicator_params()
        key = self.indicators_cache.get_indicator_key(
            name=self.name,
            params=params,
            asset_name=asset_name,
            data_length=len(df)
        )
        
        self.indicators_cache.save(
            key=key,
            df_before=df_before,
            df_after=df_result,
            name=self.name,
            params=params,
            asset_name=asset_name
        )
        
        return df_result
        
    def _get_indicator_params(self) -> Dict[str, Any]:
        """Retorna parâmetros relevantes para cache de indicadores"""
        return {
            'timeframe': self.time_rules.execution_timeframe,
            # Adicionar outros parâmetros específicos da estratégia
        }
        
    def generate_signals(self, asset_name: str = None) -> pd.DataFrame:
        """
        Gera sinais para um ativo específico ou todos do portfólio
        
        Args:
            asset_name: Nome do ativo específico ou None para todos
        """
        if asset_name:
            if asset_name not in self.asset_portfolio.assets:
                raise ValueError(f"Asset {asset_name} not found in portfolio")
            assets_to_process = {asset_name: self.asset_portfolio.assets[asset_name]}
        else:
            assets_to_process = self.asset_portfolio.assets
            
        results = {}
        for name, asset in assets_to_process.items():
            # Obtém dados do ativo
            df = asset.data_get(self.time_rules.execution_timeframe)
            if df is None or df.empty:
                continue
                
            # Tenta carregar indicadores do cache
            params = self._get_indicator_params()
            cache_key = self.indicators_cache.get_indicator_key(
                name=self.name,
                params=params,
                asset_name=name,
                data_length=len(df)
            )
            
            cached_indicators = self.indicators_cache.load(cache_key)
            if cached_indicators is not None:
                # Usa indicadores do cache
                df = df.join(cached_indicators)
            else:
                # Calcula e cache indicadores
                df = self.calculate_indicators(df, name)
            
            # Gera sinais
            df = self._apply_entry_rules(df)
            df = self._apply_exit_rules(df)
            
            # Adiciona colunas relevantes para backtest
            df = self._prepare_backtest_columns(df)
            
            results[name] = df
            
            # Print para verificação
            print(f"\nPreview do DataFrame para {name}:")
            print("\nColunas relevantes para backtest:")
            relevant_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 
                           'entry_long', 'entry_short', 'exit_long', 'exit_short',
                           'sl_long', 'sl_short', 'tp_long', 'tp_short']
            print(df[relevant_cols].head())
            
            print("\nTodas as colunas:")
            print(df.head())
            
        return results if len(assets_to_process) > 1 else next(iter(results.values()))
        
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
        




class IndicatorsCache:
    """Gerencia o cache de indicadores técnicos"""
    
    def __init__(self, db_path: str = "indicators.db"):
        self.db_path = db_path
        self._setup_db()
    
    def _setup_db(self):
        """Configura o banco de dados SQLite"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS indicators_meta
                    (indicator_id TEXT PRIMARY KEY,
                     name TEXT,
                     parameters TEXT,
                     asset_name TEXT, 
                     data_length INTEGER,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS indicators_data
                    (indicator_id TEXT,
                     column_name TEXT,
                     data BLOB,
                     PRIMARY KEY (indicator_id, column_name),
                     FOREIGN KEY (indicator_id) REFERENCES indicators_meta(indicator_id))''')
        
        conn.commit()
        conn.close()

    def get_indicator_key(self, name: str, params: Dict[str, Any], asset_name: str, data_length: int) -> str:
        """Gera chave única para o indicador"""
        params_str = json.dumps(params, sort_keys=True)
        return f"{name}_{params_str}_{asset_name}_{data_length}"

    def load(self, key: str) -> Optional[pd.DataFrame]:
        """Carrega indicador do cache"""
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("SELECT 1 FROM indicators_meta WHERE indicator_id = ?", (key,))
            if not c.fetchone():
                return None
            
            c.execute("SELECT column_name, data FROM indicators_data WHERE indicator_id = ?", (key,))
            data = {}
            for col_name, blob in c.fetchall():
                data[col_name] = pd.read_msgpack(blob)
            
            return pd.DataFrame(data)
        finally:
            conn.close()

    def save(self, key: str, df_before: pd.DataFrame, df_after: pd.DataFrame, 
             name: str, params: Dict[str, Any], asset_name: str):
        """
        Salva apenas as novas colunas criadas pelo indicador
        
        Args:
            df_before: DataFrame antes do cálculo do indicador
            df_after: DataFrame após o cálculo do indicador
        """
        # Identifica apenas as novas colunas
        new_cols = df_after.columns.difference(df_before.columns)
        if new_cols.empty:
            return
            
        new_data = df_after[new_cols]
        
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            
            c.execute("""INSERT OR REPLACE INTO indicators_meta 
                        (indicator_id, name, parameters, asset_name, data_length)
                        VALUES (?, ?, ?, ?, ?)""",
                     (key, name, json.dumps(params), asset_name, len(df_after)))
            
            for col in new_cols:
                c.execute("""INSERT OR REPLACE INTO indicators_data
                            (indicator_id, column_name, data)
                            VALUES (?, ?, ?)""",
                         (key, col, new_data[col].to_msgpack()))
            
            conn.commit()
        finally:
            conn.close()

class BacktestResult:
    """Armazena os resultados de um único backtest"""
    def __init__(self):
        self.trades: List[Trade] = []
        self.metrics: Dict[str, float] = {}
        self.parameters: Dict[str, Any] = {}

class Backtest:
    """Nova implementação do Backtest com foco em performance"""
    
    def __init__(self, stateless: bool = True):
        self.stateless = stateless
        self._open_trades: List[Trade] = []
        
    def run(self, df: pd.DataFrame, strategy: Strat) -> BacktestResult:
        """
        Executa um único backtest
        
        Args:
            df: DataFrame com dados e sinais
            strategy: Instância de Strat que gerou os sinais
        """
        result = BacktestResult()
        result.parameters = {
            'timeframe': strategy.time_rules.execution_timeframe,
            'risk': strategy.risk_rules.trade_risk_default
        }
        
        # Reset estado se necessário
        if self.stateless:
            self._open_trades = []
            
        # Itera sobre as linhas do DataFrame
        for i in range(1, len(df)):
            current_bar = df.iloc[i]
            prev_bar = df.iloc[i-1]
            
            # Processa trades abertos
            self._process_open_trades(current_bar, prev_bar, result.trades)
            
            # Verifica novas entradas
            if current_bar['entry_long']:
                self._create_trade(current_bar, strategy, 'long', result.trades)
            elif current_bar['entry_short']:
                self._create_trade(current_bar, strategy, 'short', result.trades)
                
        # Fecha trades abertos no final do período
        self._close_all_trades(df.iloc[-1], result.trades)
        
        return result
        
    def _process_open_trades(self, current_bar: pd.Series, prev_bar: pd.Series, trades: List[Trade]):
        """Processa trades abertos para saídas"""
        for trade in self._open_trades[:]:  # Copia lista para permitir remoção
            # Verifica condições de saída
            close_params = self._check_exit_conditions(trade, current_bar, prev_bar)
            if close_params:
                trade.close(close_params)
                trades.append(trade)
                self._open_trades.remove(trade)
                
    def _check_exit_conditions(self, trade: Trade, current_bar: pd.Series, prev_bar: pd.Series) -> Optional[Dict]:
        """
        Verifica condições de saída para um trade
        Retorna parâmetros de fechamento se deve fechar, None caso contrário
        """
        # Saída por Stop Loss
        if trade.direction == 'long':
            if current_bar['low'] <= trade.sl:
                return {
                    'exit_price': trade.sl,
                    'exit_type': 'sl',
                    'exit_time': current_bar['datetime']
                }
        else:  # short
            if current_bar['high'] >= trade.sl:
                return {
                    'exit_price': trade.sl,
                    'exit_type': 'sl',
                    'exit_time': current_bar['datetime']
                }
                
        # Saída por Take Profit
        if trade.direction == 'long':
            if current_bar['high'] >= trade.tp:
                return {
                    'exit_price': trade.tp,
                    'exit_type': 'tp',
                    'exit_time': current_bar['datetime']
                }
        else:  # short
            if current_bar['low'] <= trade.tp:
                return {
                    'exit_price': trade.tp,
                    'exit_type': 'tp',
                    'exit_time': current_bar['datetime']
                }
                
        # Saída por Time Frame
        if current_bar['exit_tf']:
            return {
                'exit_price': current_bar['close'],
                'exit_type': 'tf',
                'exit_time': current_bar['datetime']
            }
            
        return None
        
    def _create_trade(self, bar: pd.Series, strategy: Strat, direction: str, trades: List[Trade]):
        """Cria um novo trade"""
        # Calcula parâmetros do trade
        entry_price = bar['entry_price_long'] if direction == 'long' else bar['entry_price_short']
        sl = bar['sl_long'] if direction == 'long' else bar['sl_short']
        tp = bar['tp_long'] if direction == 'long' else bar['tp_short']
        
        # Cria trade usando a classe Trade existente
        trade_params = {
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': bar['datetime'],
            'sl': sl,
            'tp': tp,
            'size': 1.0,  # Será calculado pelo Money Management
            'asset': bar['ativo']
        }
        
        trade = Trade(trade_params)
        self._open_trades.append(trade)
        
    def _close_all_trades(self, last_bar: pd.Series, trades: List[Trade]):
        """Fecha todos os trades abertos"""
        for trade in self._open_trades[:]:
            close_params = {
                'exit_price': last_bar['close'],
                'exit_type': 'eod',  # end of data
                'exit_time': last_bar['datetime']
            }
            trade.close(close_params)
            trades.append(trade)
        self._open_trades.clear()

def run_backtest(self, strategy_name: str, asset_name: str = None, stateless: bool = True) -> Optional[Dict[str, Any]]:
    """
    Executa backtest para uma estratégia
    
    Args:
        strategy_name: Nome da estratégia
        asset_name: Nome do ativo específico ou None para todos
        stateless: Se True, não mantém estado entre execuções
    """
    try:
        # Carrega a estratégia
        strategy = self.read(Strat, strategy_name)
        if not strategy:
            print(f"Estratégia {strategy_name} não encontrada")
            return None
            
        # Gera/atualiza sinais
        signals = strategy.generate_signals(asset_name)
        if not signals:
            return None
            
        # Cria instância do Backtest
        backtest = Backtest(stateless=stateless)
        
        # Executa backtest para cada ativo
        results = {}
        if isinstance(signals, dict):
            for name, df in signals.items():
                print(f"\nExecutando backtest para {name}...")
                result = backtest.run(df, strategy)
                results[name] = result
        else:
            name = asset_name or next(iter(strategy.asset_portfolio.assets.keys()))
            print(f"\nExecutando backtest para {name}...")
            results[name] = backtest.run(signals, strategy)
            
        return results
        
    except Exception as e:
        print(f"Erro ao executar backtest: {e}")
        return None

def run_backtest_setup(self, strat_name: str, portfolio_name: str = None, asset_name: str = None) -> bool:
    """
    Prepara e executa um backtest para uma estratégia
    
    Args:
        strat_name: Nome da estratégia
        portfolio_name: Nome do portfólio (opcional)
        asset_name: Nome do ativo específico no portfólio (opcional)
    """
    try:
        # 1. Carrega a estratégia
        strategy = self.read(Strat, strat_name)
        if not strategy:
            print(f"Estratégia {strat_name} não encontrada")
            return False
            
        # 2. Carrega/cria portfólio
        if portfolio_name:
            portfolio = self.read(Asset_Portfolio, portfolio_name)
            if not portfolio:
                print(f"Portfólio {portfolio_name} não encontrado")
                return False
        else:
            # Se não foi especificado um portfólio, verifica se a estratégia já tem um
            if hasattr(strategy, 'asset_portfolio') and strategy.asset_portfolio:
                portfolio = strategy.asset_portfolio
            else:
                print("Nenhum portfólio especificado ou associado à estratégia")
                return False
                
        # 3. Verifica/carrega Money Management
        mm = None
        if hasattr(strategy, 'risk_rules'):
            # Cria um MM genérico baseado nas regras de risco da estratégia
            mm = Money_Management_Algorithm({
                'name': f"{strat_name}_mm",
                'position_sizing_type': strategy.risk_rules.position_sizing_type,
                'position_sizing_from': strategy.risk_rules.position_sizing_from,
                'position_sizing_method': strategy.risk_rules.position_sizing_method,
                'trade_risk_default': strategy.risk_rules.trade_risk_default,
                'trade_risk_min': strategy.risk_rules.trade_risk_min,
                'trade_risk_max': strategy.risk_rules.trade_risk_max
            })
            
        # 4. Gera sinais
        if asset_name:
            # Para um ativo específico
            if asset_name not in portfolio.assets:
                print(f"Ativo {asset_name} não encontrado no portfólio")
                return False
            df = strategy.generate_signals(asset_name)
            
            # 5. Mostra o DataFrame com os sinais
            print(f"\nDataFrame com sinais para {asset_name}:")
            print("\nPrimeiras linhas:")
            print(df.head())
            print("\nÚltimas linhas:")
            print(df.tail())
            
            # Mostra colunas relevantes
            signal_cols = [col for col in df.columns if 'signal' in col.lower()]
            if signal_cols:
                print("\nColunas de sinais:")
                print(df[signal_cols].describe())
                
        else:
            # Para todos os ativos no portfólio
            for curr_asset_name in portfolio.assets:
                df = strategy.generate_signals(curr_asset_name)
                
                print(f"\nDataFrame com sinais para {curr_asset_name}:")
                print("\nPrimeiras linhas:")
                print(df.head())
                print("\nÚltimas linhas:")
                print(df.tail())
                
                # Mostra colunas relevantes
                signal_cols = [col for col in df.columns if 'signal' in col.lower()]
                if signal_cols:
                    print("\nColunas de sinais:")
                    print(df[signal_cols].describe())
                    
        return True
        
    except Exception as e:
        print(f"Erro ao preparar backtest: {e}")
        import traceback
        traceback.print_exc()
        return False








  




