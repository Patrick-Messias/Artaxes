from typing import Union, Dict, Optional
from dataclasses import dataclass, field
import BaseClass, Persistance, uuid
import pandas as pd
from Model import Model
from Portfolio import Portfolio
from Backtest import Backtest
from Optimization import Optimization
from Walkforward import Walkforward
from Asset import Asset, Asset_Portfolio

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
    name: str = field(default_factory=lambda: f'model_{uuid.uuid4()}')
    data: Union[Model, Portfolio]=None # Can make an operation with a single model or portfolio
    operation: Union[Backtest, Optimization, Walkforward]=None 

    # Metrics
    metrics: Optional[Dict[str, Indicator]] = field(default_factory=dict)

    # Settings
    operation_timeframe: str=None
    date_start: str=None
    date_end: str=None
    save: bool=False
    
class Operation(BaseClass, Persistance):
    def __init__(self, op_params: Operation_Parameters):
        super().__init__()
        self.name = op_params.name
        self.data = op_params.data
        self.operation = op_params.operation

        self.metrics = op_params.metrics

        self.operation_timeframe = op_params.operation_timeframe
        self.date_start = op_params.date_start
        self.date_end = op_params.date_end
        self.save = op_params.save

        # Caches organize and save data for general use in various defs
        self._assets_data_cache = {}        # {(asset_name, timeframe)}
        self._indicators_cache = {}         # {(ind_name, asset_name, timeframe, params)}
        self._signal_cache = {}             # {(model_name, strat_name, asset_name, timeframe, params)}
        self._preliminary_result_cache = {} # {(model_name, strat_name, asset_name, timeframe, params)}

        self._operation_result = {}         # {(model_name, strat_name, asset_name)}

    def _get_all_models(self) -> dict: # Returns all Model(s) from data
        if isinstance(self.data, Model): # Single Model
            return {self.data.name: self.data}
        elif isinstance(self.data, Portfolio): # Portfolio
            return self.data._get_all_models()
        else: return {}

    def _get_all_assets(self) -> dict: # Returns all Asset(s) from Models
        models = self._get_all_models()
        assets={}
        for name, model in models.items():
            if isinstance(model.assets, Asset): # Asset
                if model.assets.name not in assets: 
                    assets[model.assets.name] = model.assets

            elif isinstance(model.assets, Asset_Portfolio): # Asset_Portfolio
                for asset_name, asset in model.assets.assets.items():
                    if asset_name not in assets:
                        assets[asset_name] = asset

            else: print(f"⚠️ Warning: model '{name}' has no valid assets (neither Asset nor Asset_Portfolio)")

        return assets

    def _get_all_strats(self) -> dict: # Returns all Strat(s) from Models
        models = self._get_all_models()
        strats={}
        for name, model in models.items():
            strats[name] = model.strat
        return strats

    def _get_all_unique_datetimes(self, assets_cache=None): # Returns all unique datetimes from a Asset dict
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

        return sorted(all_unique_datetimes)



    def _validate_operation(self):
        """
        Validates the operation configuration before execution.
        Raises ValueError with descriptive message if validation fails.
        """
        errors = []
        warnings = []
        
        # 1. Validate basic operation structure
        if not self.data:
            errors.append("❌ Operation data (Model/Portfolio) is required")
        
        if not self.operation:
            errors.append("❌ Operation type (Backtest/Optimization/Walkforward) is required")
        
        # 2. Validate data structure
        if self.data:
            if not isinstance(self.data, (Model, Portfolio)):
                errors.append("❌ Data must be either Model or Portfolio instance")
            
            # Validate models exist and have required components
            models = self._get_all_models()
            if not models:
                errors.append("❌ No models found in data")
            
            for model_name, model in models.items():
                # Validate model has strategies
                if not hasattr(model, 'strat') or not model.strat:
                    errors.append(f"❌ Model '{model_name}' has no strategies")
                
                # Validate model has assets
                if not hasattr(model, 'assets') or not model.assets:
                    errors.append(f"❌ Model '{model_name}' has no assets")
                
                # Validate strategies within model
                if hasattr(model, 'strat') and model.strat:
                    for strat_name, strat in model.strat.items():
                        # Validate strategy has required components
                        required_attrs = ['name', 'asset_mapping', 'indicators', 'entry_rules']
                        for attr in required_attrs:
                            if not hasattr(strat, attr):
                                errors.append(f"❌ Strategy '{strat_name}' missing required attribute: {attr}")
                        
                        # Validate asset mapping
                        if hasattr(strat, 'asset_mapping') and strat.asset_mapping:
                            if not isinstance(strat.asset_mapping, dict):
                                errors.append(f"❌ Strategy '{strat_name}' asset_mapping must be a dictionary")
                            else:
                                for asset_key, asset_config in strat.asset_mapping.items():
                                    if not isinstance(asset_config, dict):
                                        errors.append(f"❌ Strategy '{strat_name}' asset_mapping['{asset_key}'] must be a dictionary")
                                    elif 'name' not in asset_config or 'timeframe' not in asset_config:
                                        errors.append(f"❌ Strategy '{strat_name}' asset_mapping['{asset_key}'] missing 'name' or 'timeframe'")
                        
                        # Validate indicators
                        if hasattr(strat, 'indicators') and strat.indicators:
                            if not isinstance(strat.indicators, dict):
                                errors.append(f"❌ Strategy '{strat_name}' indicators must be a dictionary")
                        
                        # Validate entry rules
                        if hasattr(strat, 'entry_rules') and strat.entry_rules:
                            if not isinstance(strat.entry_rules, dict):
                                errors.append(f"❌ Strategy '{strat_name}' entry_rules must be a dictionary")
                            elif not any(key in strat.entry_rules for key in ['entry_long', 'entry_short']):
                                warnings.append(f"⚠️ Strategy '{strat_name}' has no entry_long or entry_short rules")
        
        # 3. Validate operation type
        if self.operation:
            valid_operation_types = (Backtest, Optimization, Walkforward)
            if not isinstance(self.operation, valid_operation_types):
                errors.append("❌ Operation must be Backtest, Optimization, or Walkforward instance")
        
        # 4. Validate timeframes and dates
        if self.operation_timeframe and not isinstance(self.operation_timeframe, str):
            errors.append("❌ operation_timeframe must be a string")
        
        if self.date_start and not isinstance(self.date_start, str):
            warnings.append("⚠️ date_start should be a string in YYYY-MM-DD format")
        
        if self.date_end and not isinstance(self.date_end, str):
            warnings.append("⚠️ date_end should be a string in YYYY-MM-DD format")
        
        # 5. Validate metrics configuration
        if self.metrics:
            if not isinstance(self.metrics, dict):
                errors.append("❌ metrics must be a dictionary")
        
        # 6. Check for common configuration issues
        if self.data and isinstance(self.data, Portfolio):
            if not hasattr(self.data, 'models') or not self.data.models:
                warnings.append("⚠️ Portfolio has no models defined")
        
        # Report validation results
        if warnings:
            for warning in warnings:
                print(warning)
        
        if errors:
            error_message = "Operation validation failed:\n" + "\n".join(errors)
            raise ValueError(error_message)
        
        print("✅ Operation validation passed")
        return True



    def _calculates_indicators(self): # Calculates all Indicators for all Models with their own Assets, saves to _indicators_cache
        models = self._get_all_models()

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
                        if (asset.name, ind.timeframe) in self._assets_data_cache:
                            data = self._assets_data_cache[(asset.name, ind.timeframe)] # Saves ind.timeframe because it is the timeframe the ind will use, usually the Strats tf
                        else: 
                            data = asset.data_get(ind.timeframe)
                            self._assets_data_cache[(asset.name, ind.timeframe)] = data
                        
                        for param_name, param_value in ind.params.items():
                            cache_key = (ind.name, asset.name, ind.timeframe, param_value)

                            if cache_key not in self._indicators_cache:
                                self._indicators_cache[cache_key] = ind.calculate_indicator(data, param_value)
        return None

    def _generate_signals(self):
        return None

    def _preliminary_backtest(self):
        return None


    def _calculate_metrics():
        return None

    def _save_results():
        return None

    def run(self):

        # I - Init and Validation
        self._validate_operation()

        # II - Data Pre-Processing
        self._calculates_indicators()
        self._generate_signals()
        self._preliminary_backtest() # If simple backtest then stops here?

        # III - Execution
        if isinstance(self.operation, Backtest): self._execute_backtest()
        elif isinstance(self.operation, Optimization): self._execute_optimization()
        elif isinstance(self.operation, Walkforward): self._execute_walkforward()

        # IV - Pos-Processing
        if self.metrics: self._calculate_metrics()
        if self.save: self._save_results()

        return self._operation_result





    