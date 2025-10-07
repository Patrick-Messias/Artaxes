from typing import Union, Dict, Optional
from dataclasses import dataclass, field
import BaseClass

# NOTE -> Tirar daqui -> 
# 1. Criar um HMM para modelar transição de estados de volatilidade [high, med, low]. 
# Then add a latent variable like trend (bull, sideways, bear) to add new states [high_bull, high_sideways, etc.]
# 2. Criar uma def que roda vários Assets e identifica o HMM mais adequado para aquele grupo de Assets

# CONFIGURAR TUDO, PRINCIPALMENTE MM, TM e SM para JSON para otimização e evitar recompilação

""" # Order of Execution
Operation:

    1. Data Preload
        - loads all unique data to memory cache
        - calculates all unique Indicators for all Strats
        - generates all signals for all Strats
        - runs preliminary backtest with all param combinations and saves to use for WF

    2. Operation Execution
    for datetime in all_unique_datetimes:
        
        # 1. Atualizar posições abertas
        for trade in trades:
            - check_exit_signals (tf/sl/tp/breakeven/etc.)
        
        # 2. Gerar sinais de entrada
        for asset in all_unique_assets:
            for model in models:
                for strat in model.strats:
                    - generate_entry_signals (long/short/opcionalmente com ranking)
        
        # 3. Aplicar regras de portfólio (PMA)
        - portfolio_rebalancing (definir o que realmente entra/sai do portfólio)
        
        # 4. Executar ordens
        - execute_orders (abrir/fechar trades com base nos sinais + rebalanceamento)
"""

@dataclass
class Operation_Parameters():
    name: str='unnamed_operation'
    data: Union[Model, Portfolio]=None # Can make an operation with a single model or portfolio
    operation: Union[Backtest, Optimization, Walkforward]=None 

    # Metrics
    metrics: Optional[Dict[str, Indicator]] = field(default_factory=dict)

    # Settings
    operation_timeframe: str=None
    date_start: str=None
    date_end: str=None
    
class Operation(BaseClass):
    def __init__(self, op_params: Operation_Parameters):
        super().__init__()
        self.name = op_params.name
        self.data = op_params.data
        self.operation = op_params.operation
        self.metrics = op_params.metrics
        self.operation_timeframe = op_params.operation_timeframe
        self.date_start = op_params.date_start
        self.date_end = op_params.date_end

        Acho naõ ser necessário o operation_map, só usar os caches para se organizar #self._operation_map = {}

        # Caches organize and save data for general use in various defs
        self._assets_data_cache = {}        # {(asset_name, timeframe)}
        self._indicators_cache = {}         # {(ind_name, asset_name, timeframe, params)}
        self._signal_cache = {}             # {(model_name, strat_name, asset_name, timeframe, params)}
        self._preliminary_result_cache = {} # {(model_name, strat_name, asset_name, timeframe, params)}

        self._operation_result = {}         # {(model_name, strat_name, asset_name)}

    def get_all_models(self) -> dict: # Returns all Model(s) from data
        if isinstance(self.data, Model): # Single Model
            return {self.data.name: self.data}
        elif isinstance(self.data, Portfolio): # Portfolio
            return self.data.get_all_models()
        else: return {}

    def get_all_assets(self) -> dict: # Returns all Asset(s) from Models
        models = self.get_all_models()
        assets={}
        for name, model in models.items():
            if isinstance(model.assets, Asset): # Asset
                if model.assets.name not in assets: 
                    assets[model.assets.name] = model.assets

            elif isistance(model.assets, Asset_Portfolio): # Asset_Portfolio
                for asset_name, asset in model.assets.assets.items():
                    if asset_name not in assets:
                        assets[asset_name] = asset

            else: print(f"⚠️ Warning: model '{name}' has no valid assets (neither Asset nor Asset_Portfolio)")

        return assets

    def get_all_strats(self) -> dict: # Returns all Strat(s) from Models
        models = self.get_all_models()
        strats={}
        for name, model in models.items():
            strats[name] = model.strat
        return strats

    def get_all_unique_datetimes(self, assets_cache=None): # Returns all unique datetimes from a Asset dict
        all_unique_datetimes = set()

        if not assets_cache: return []

        for (asset_name, tf), df in assets_cache.items():
            if df is None or df.empty: continue

            if 'datetime' in df.columns:
                datetimes = pd.to_datetime(df['datetime'], errors='coerse')
            elif 'date' in df.columns:
                datetimes = pd.to_datetime(df['date'], errors='coerse')
            else: 
                continue

            datetimes = datetimes.dt.normalize()
            all_unique_datetimes.update(datetimes.dropna().unique())

        return sorted(all_unique_datetime)

    def calculates_indicators(self): # Calculates all Indicators for all Models with their own Assets, saves to _indicators_cache
        models = self.get_all_models()

        for name, model in models.items():
            # strats = model.strat
            assets = {}

            if isinstance(model.assets, Asset): 
                assets[model.assets.name] = model.assets
            elif isinstance(model.assets, Asset_Portfolio): 
                for asset_name, asset in model.assets.assets.items(): 
                    assets[asset_name] = asset
            else: print(f"⚠️ Warning: model has no valid assets (neither Asset nor Asset_Portfolio)")

            for asset_name, asset in assets.items():

                # 1. Add exception if operational_timeframe != ind.timeframe, OR add this later in signals

                for strat in strats:
                    for ind in strat.indicators:
                        if (asset.name, ind.timeframe) in _assets_data_cache:
                            data = _assets_data_cache[(asset.name, ind.timeframe)] # Saves ind.timeframe because it is the timeframe the ind will use, usually the Strats tf
                        else: 
                            data = asset.data_get(ind.timeframe)
                            _assets_data_cache[(asset.name, ind.timeframe)] = data
                        
                        for param_name, param_value in ind.params.items():
                            cache_key = (ind.name, asset.name, ind.timeframe, param_value)

                            if cache_key not in self._indicators_cache:
                                self._indicators_cache[cache_key] = ind.calculate_indicator(data, param_value)
        return None

    def generate_signals(self):
        return None

    def preload_data(self):
        return None

    def preliminary_backtest(self):
        return None

    def run(self):
        return None





    