from typing import Union, Dict, Optional, Any
from dataclasses import dataclass, field
import BaseClass, Persistance, uuid
import pandas as pd
from Model import Model
from Portfolio import Portfolio
from Backtest import Backtest
from Optimization import Optimization
from Walkforward import Walkforward
from Asset import Asset, Asset_Portfolio
from Persistance import save, load
from OptimizedOperationResult import OptimizedOperationResult

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
        #self._assets_data_cache = {}        # {(asset_name, timeframe)}
        #self._indicators_cache = {}         # {(ind_name, asset_name, timeframe, params)}
        #self._signal_cache = {}             # {(model_name, strat_name, asset_name, timeframe, params)}
        #self._preliminary_result_cache = {} # {(model_name, strat_name, asset_name, timeframe, params)}

        # Cache otimizado
        self._memory_cache = {}
        self._cache_size_limit = 100 * 1024 * 1024  # 100MB limit

        self._operation_result = OptimizedOperationResult() # {(model_name, strat_name, asset_name)}

    def _get_all_models(self) -> dict: # Returns all Model(s) from data
        if isinstance(self.data, Model): # Single Model
            return {self.data.name: self.data}
        elif isinstance(self.data, Portfolio): # Portfolio
            return self.data._get_all_models()
        else: return {}

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


    # I - Init and Validation
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

    # II - Hierarchical Mapping
    def _mapping(self): # Creates an optmized hierarchical mapping of the results
        models = self._get_all_models()

        # Base Structure
        portfolio_name = getattr(self.data, 'name', 'default_portfolio')

        for model_name, model in models.items():
            model_path = f'portfolio.models.{model_name}'

            # Model's Assets
            assets = self._get_model_assets(model)
            self._operation_result.set_result(f"{model_path}.assets", assets)

            # Model's Strats
            if hasattr(model, 'strat') and model.strat:
                for strat_name, strat in model.strat.items():
                    strat_path = f"{model_path}.strats.{strat_name}"

                    # Strat's Indicators
                    if hasattr(strat, 'indicators'):
                        self._operation_result.set_result(
                            f"{strat_path}.indicators",
                            strat.indicators
                        )

                    # Strat's Asset Mapping
                    if hasattr(strat, 'asset_mapping'):
                        self._operation_result.set_result(
                            f"{strat_path}.asset_mapping",
                            strat.asset_mapping
                        )

                    # Strat's Results PLACEHOLDER
                    self._operation_result.set_result(
                        f'{strat_path}.results',
                        {}
                    )

            # Shared indicators in the model
            if hasattr(model, 'indicators'):
                self._operation_result.set_result(
                    f"{model_path}.shared_indicators", 
                    model.indicators
                )

    def _get_model_assets(self, model) -> Dict[str, Any]: # Optimally extracts assets from a model
        asset_info={}

        if isinstance(model.assets, Asset):
            assets_info[model.assets.name] = {
                'type': model.assets.type,
                'market': model.assets.market,
                'timeframes': list(model.assets.data.keys())
            }
        elif isinstance(model.assets, Asset_Portfolio):
            for asset_name, asset in model.assets.assets.items():
                assets_info[asset_name] = {
                    'type': asset.type,
                    'market': asset.market,
                    'timeframes': list(asset.data.keys())
                }
        return assets_info

    # Helper methos for fast access
    def get_model_results(self, model_name: str) -> Dict[str, Any]: # Fast access for Models results
        return self._operation_result.get_result(f"portfolio.models.{model_name}")

    def get_strat_results(self, model_name: str, strat_name: str) -> Dict[str, Any]: # Fast access for Strat results
        return self._operation_result.get_result(f"portfolio.models.{model_name}.strats.{strat_name}.results")

    def set_strat_result(self, model_name: str, strat_name: str, result_type: str, data: Any) -> None: # Defines a Strat's results
        path = f"portfolio.models.{model_name}.strats.{strat_name}.results.{result_type}"
        self._operation_result.set_result(path, data)

    def get_all_strat_results(self) -> Dict[str, Any]: # Recovers results from all Strats
        models = self._get_all_models()
        results = {}
        
        for model_name in models.keys():
            model_results = self.get_model_results(model_name)
            if 'strats' in model_results:
                results[model_name] = model_results['strats']
    
        return results

    # III - Data Pre-Processing
    def _calculates_indicators(self): # Calculates all indicators using mapped structure
        try:
            portfolio_data = self._operation_result.get_result("portfolio")
        except KeyError:
            print("❌ No portfolio mapping found. Run _mapping() first.")
            return None
        
        models = portfolio_data.get('models', {})

        for model_name, model_data in models.items(): # Processes each Strat in each Model
            strats = model_data.get('strats', {})

            for strat_name, strat_data in strats.items(): # Get any indicators and asset_mapping already mapped
                indicators = strat_data.get('indicators', {})
                asset_mapping = strat_data.get('asset_mapping', {})

                # Calculates indicators based on mapping
                self._calculate_strat_indicators(model_name, strat_name, indicators, asset_mapping)
        return None

    def _calculates_strat_indicators(self, model_name: str, strat_name: str, indicators: dict, asset_mapping: dict): # Calculates indicators for a specific Strat using mapped data
        for asset_key, asset_config in asset_mapping.items():
            asset_name = asset_config.get('name')
            timeframe = asset_config.get('timeframe')

            if not asset_name or not timeframe: continue

            # Gets the real object (not mapped metadata)
            asset_obj = self._get_asset_object(model_name, asset_name)
            if not asset_obj: continue

            # Loads Asset data
            try:
                data = asset_obj.data_get(timeframe)
                if data is None or data.empty: continue
            except Exception as e:
                print(f"⚠️ Error loading data for {asset_name}: {e}")
                continue

            # Calculates each indicator
            for ind_name, indicator in indicators.items():
                if indicator.timeframe != timeframe: continue

                # Calculates and stores results
                result_path = f"portfolio.models.{model_name}.strats.{strat_name}.results.indicators.{asset_name}.{timeframe}.{ind_name}"
                calculated_data = self._calculate_indicator_data(data, indicator)
                
                self._operation_result.set_result(result_path, calculated_data)

    def _get_asset_object(self, model_name, asset_name: str): # Gets the real Asset object (not mapped metadata)
        models = self._get_all_models()
        model = models.get(model_name)

        if not model: return None

        if isinstance(model.assets, Asset):
            return model.assets if model.assets.name == asset_name else None
        elif isinstance(model.assets, Asset_Portfolio):
            return model.assets.assets.get(asset_name)
        return None


    def _generate_signals(self):
        return None

    def _preliminary_backtest(self):
        return None

    # IV - Execution

    # V - Pos-Processing
    def _calculate_metrics():
        return None

    # VI - Saving
    def save_results(self, filepath: str=None) -> str: # Optimally saves results
        if not filepath: filepath = f"operation_results_{self.name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.gz"
        self._operation_result.save_to_file(filepath)
        return filepath
        
    def load_results(self, filepath: str) -> None: # Loads saved results
        self._operation_result = OptimizedOperationResult.load_from_file(filepath)

    # VII - Cleanup
    def cleanup_memory(self) -> None: # clears cache's memory
        if len(self._memory_cache) > self._cache_size_limit: 
            sorted_items = sorted( # Removes least necessary itens
                self._memory_cache.items(),
                key=lambda x: x[1].get('last_accessed', 0)
            )

            # Remove 50% of least used itens
            items_to_remove = len(sorted_items) // 2
            for key, _ in sorted_items[:items_to_remove]:
                del self._memory_cache[key]
    


    def run(self):

        # I - Init and Validation
        self._validate_operation()

        # II - Hierarchical Mapping
        self._mapping()

        # III - Data Pre-Processing
        self._calculates_indicators()
        self._generate_signals()
        self._preliminary_backtest() # If simple backtest then stops here?

        # IV - Execution
        if isinstance(self.operation, Backtest): self._execute_backtest()
        elif isinstance(self.operation, Optimization): self._execute_optimization()
        elif isinstance(self.operation, Walkforward): self._execute_walkforward()

        # V - Pos-Processing
        if self.metrics: self._calculate_metrics()

        # VI - Saving
        if self.save: self.save_results()

        # VII - Cleanup
        self.cleanup_memory()

        return self._operation_result



""" OLD BELOW, DELETE AFTER

  
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


"""
    