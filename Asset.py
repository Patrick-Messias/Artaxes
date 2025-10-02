import pandas as pd, os, re, import BaseClass
from dataclasses import dataclass

@staticmethod
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
        if not isinstance(name, str) or not name: raise ValueError("Invalid asset name")
        if type not in self.ASSET_PARAMS: raise ValueError(f"Asset type not found: {list(self.ASSET_PARAMS.keys())}")
        
        self.name = name
        self.type = type
        self.market = market
        self.data_path = data_path
        self.data: dict[str, pd.DataFrame] = {}
        if timeframe: 
            for tf in timeframe:
                self.data[tf] = None
        else: self.timeframes_load_available()

        if not os.path.isdir(self.data_path): print(f"⚠️ Warning: Data path '{self.data_path}' not found when initializing Asset.")

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

    @staticmethod
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

    @staticmethod
    def plot_correlation(self, timeframe: str, figsize=(12, 10)):
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        corr_matrix = self.calculate_correlation(timeframe)
        if not corr_matrix.empty:
            plt.figure(figsize=figsize)
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title(f'Correlation Matrix - {timeframe}')
            plt.show()

    def preload_data(self, timeframes: list[str], verbose: bool = False) -> None:
        """
        Precarrega os dados de todos os assets do portfólio para os timeframes especificados.
        
        Args:
            timeframes: Lista de timeframes para carregar (ex: ['M5', 'M15', 'H1'])
            verbose: Se True, imprime informações sobre o progresso do carregamento
        """
        if verbose:
            print(f"\nPrecarregando dados para {len(self.assets)} assets em {len(timeframes)} timeframes...")
        
        for asset_name, asset in self.assets.items():
            if verbose:
                print(f"\nCarregando dados para {asset_name}:")
            
            for tf in timeframes:
                try:
                    if verbose:
                        print(f"  - Timeframe {tf}... ", end='')
                    df = asset.data_get(tf)
                    if verbose:
                        print(f"OK ({len(df)} linhas)")
                except Exception as e:
                    if verbose:
                        print(f"ERRO: {str(e)}")

    def __repr__(self):
        return self.__str__()



    """
    def preload_strategies(self, strategies: list["Strat"], verbose: bool = False) -> None: NEEDS REDO preload_strategies ELIMINATED 29092025
        
        #Precarrega os dados necessários para uma lista de estratégias.
        #
        #Args:
        #    strategies: Lista de estratégias para precarregar dados
        #    verbose: Se True, imprime informações sobre o progresso do carregamento
        
        if verbose:
            print(f"\nPrecarregando dados para {len(strategies)} estratégias...")
        
        # Coleta todos os timeframes únicos necessários
        timeframes = set()
        for strat in strategies:
            # Timeframe principal da estratégia
            if hasattr(strat, 'time_rules'):
                timeframes.add(strat.time_rules.execution_timeframe)
            # Timeframes adicionais da estratégia
            if hasattr(strat, 'data') and hasattr(strat.data, 'additional_timeframes'):
                timeframes.update(strat.data.additional_timeframes)
        
        # Precarrega os dados para todos os timeframes necessários
        self.preload_data(list(timeframes), verbose)
    """

