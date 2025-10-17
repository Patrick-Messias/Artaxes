from typing import Union, Dict, List, Optional, Any
from dataclasses import dataclass, field
from Model import Model
from BaseClass import BaseClass
from Portfolio import Portfolio
from Backtest import Backtest
from Indicator import Indicator
from Optimization import Optimization
from Walkforward import Walkforward
from Asset import Asset, Asset_Portfolio, create_datetime_columns #, assets_info
from Persistance import Persistance
from OptimizedOperationResult import OptimizedOperationResult
import uuid, copy
import pandas as pd

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
class OperationParams():
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
    def __init__(self, op_params: OperationParams):
        super().__init__()
        self.name = op_params.name
        self.data = op_params.data
        self.operation = op_params.operation

        self.metrics = op_params.metrics

        self.operation_timeframe = op_params.operation_timeframe
        self.date_start = op_params.date_start
        self.date_end = op_params.date_end
        self.save = op_params.save

        # Optimized Cache
        self._memory_cache = {}
        self._cache_size_limit = 100 * 1024 * 1024  # 100MB limit

        self._operation_result = OptimizedOperationResult() # {(model_name, strat_name, asset_name)}

    def _get_all_models(self) -> dict: # Returns all Model(s) from data
        if isinstance(self.data, Model): # Single Model
            return {self.data.name: self.data}
        elif isinstance(self.data, Portfolio): # Portfolio
            return self.data.get_all_models()
        else: return {}



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
                        required_attrs = ['name', 'indicators', 'entry_rules']
                        for attr in required_attrs:
                            if not hasattr(strat, attr):
                                errors.append(f"❌ Strategy '{strat_name}' missing required attribute: {attr}")
                        
                        # Validate asset mapping
                        if hasattr(strat, 'strat_support_assets') and strat.strat_support_assets:
                            if not isinstance(strat.strat_support_assets, dict):
                                errors.append(f"❌ Strategy '{strat_name}' strat_support_assets must be a dictionary")
                        
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
            print(f"-> Mapped model(s): '{model_name}'")

            # Model's Assets
            assets = self._get_model_assets(model)
            self._operation_result.set_result(f"{model_path}.assets", assets)
            for asset_name, asset_obj in assets.items(): print(f"-> Mapped asset(s): '{asset_obj.name}' | with timeframe(s): {asset_obj.timeframe}")

            # Model's Strats
            if hasattr(model, 'strat') and model.strat:
                for strat_name, strat in model.strat.items():
                    strat_path = f"{model_path}.strats.{strat_name}"

                    print(f"-> Mapped strat: '{strat_name}' for model: '{model_name}'")
                    
                    # Strat's Indicators
                    if hasattr(strat, 'indicators'):
                        for ind_name, indicator in strat.indicators.items():
                            print(f"-> Mapped indicator: '{ind_name}' for strat: '{strat_name}'")
                        self._operation_result.set_result(
                            f"{strat_path}.indicators",
                            strat.indicators
                        )
                     
                    # Strat's Support Assets
                    if hasattr(strat, 'strat_support_assets'):
                        for asset_name, _ in strat.strat_support_assets.items():
                            print(f"-> Mapped strat_support_assets: '{asset_name}'")
                        self._operation_result.set_result(
                            f"{strat_path}.strat_support_assets",
                            strat.strat_support_assets
                        )

                    # Strat's Results PLACEHOLDER
                    self._operation_result.set_result(
                        f'{strat_path}.results',
                        {}
                    )

            # DELETED BECAUSE STRAT.INDICATORS != MODEL.INDICATORS | SEE IF WANT TO KEEP HERE OR MIGRATE TO MSM
            # # Shared indicators in the model 
            # if hasattr(model, 'indicators'):
            #     self._operation_result.set_result(
            #         f"{model_path}.shared_indicators", 
            #         model.indicators
            #     )

    def _get_model_assets(self, model) -> Dict[str, Any]: # Optimally extracts assets from a model
        assets={}
        if isinstance(model.assets, Asset):
            assets[model.assets.name] = model.assets
        elif isinstance(model.assets, Asset_Portfolio):
            for asset_name, asset in model.assets.assets.items():
                assets[asset_name] = asset
        return assets
        # if isinstance(model.assets, Asset):
        #     assets_info[model.assets.name] = {
        #         'data': model.assets.data,
        #         'timeframes': list(model.assets.data.keys())
        #     }
        # elif isinstance(model.assets, Asset_Portfolio):
        #     for asset_name, asset in model.assets.assets.items():
        #         assets_info[asset_name] = {
        #             'data': asset.data,
        #             'timeframes': list(asset.data.keys())
        #         }
        # return assets_info

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
            print(f"> {model_name}")
            strats = model_data.get('strats', {})
            model_assets = model_data.get('assets', {})

            for strat_name, strat_data in strats.items(): # Get any indicators and strat_support_assets already mapped
                indicators = strat_data.get('indicators', {})
                strat_support_assets = strat_data.get('strat_support_assets', {})
                
                # Calculates indicators based on mapping
                for ind_key, ind_obj in indicators.items():
                    print(f"    > asset(s): {ind_obj.asset} | strat: {strat_name} | indicator: {ind_key}")

                    if ind_obj.asset == 'CURR_ASSET': # Calcula com cada asset de model_asset
                        self._calculate_strat_indicator(model_name, strat_name, ind_key, ind_obj, model_assets)
                    else: # Calcula ind apenas com o ind_obj.asset que esteja em strat_support_assets
                        self._calculate_strat_indicator(model_name, strat_name, ind_key, ind_obj, strat_support_assets, ind_obj.asset)
        return None

    def _calculate_strat_indicator(self, model_name: str, strat_name: str, ind_name: str, ind_obj, assets: dict, asset_name=None): # Calculates indicators for a specific Strat using mapped data
        ind_asset_obj_all = {}
        if asset_name is None: # Calcula ind para cada Asset em Model
            for asset_key, asset_obj in assets.items():
                asset_obj = self._get_asset_object(model_name, asset_key)
                result_path = f"portfolio.models.{model_name}.strats.{strat_name}.results.indicators.{asset_obj.name}.{ind_obj.timeframe}.{ind_name}"
                ind_asset_obj_all[result_path] = asset_obj
        else:   # Calcula apenas para o Asset asset_name
            asset_obj = assets[asset_name]
            result_path = f"portfolio.models.{model_name}.strats.{strat_name}.results.indicators.{asset_obj.name}.{ind_obj.timeframe}.{ind_name}"
            ind_asset_obj_all[result_path] = asset_obj

        for result_path, asset_obj in ind_asset_obj_all.items():
            try: # Loads Asset data
                data = asset_obj.data_get(ind_obj.timeframe)
                if data is None or data.empty: continue
            except Exception as e:
                print(f"⚠️ Error loading data for {asset_name}: {e}")
                continue

            # Calcula o indicador com ind.calculate
            calculated_data = ind_obj.calculate_all_sets(data, base_path=result_path) # Ind params are separated by "_" suffix

            # Salva no Mapping
            for name, ind in calculated_data.items():
                print(name)
                self._operation_result.set_result(name, ind)



    def _generate_signals(self): 
        portfolio_data = self._operation_result.get_result("portfolio")
        models = portfolio_data.get('models', {})

        for model_name, model_data in models.items(): # Processes each Strat in each Model
            print(f"        > {model_name}")
            strats = model_data.get('strats', {})
            model_assets = model_data.get('assets', {})

            # Variable saves any unique timeframe != operation_timeframe for all relevant model.assets and strat.strat_support_assets
            unique_timeframes_model = {}

            for asset_name, asset_obj in model_assets.items():
                timeframes = getattr(asset_obj, "timeframe", [])
                if not isinstance(timeframes, (list, tuple, set)): timeframes = [timeframes]  # Garantees that timeframes is iterable
                for tf in timeframes:
                    if tf and tf != self.operation_timeframe:
                        unique_timeframes_model.setdefault(asset_name, []).append(tf) # cria lista se ainda não existir

            for strat_name, strat_data in strats.items(): 
                unique_timeframes_all = copy.deepcopy(unique_timeframes_model) # Reseta a comparação a cada Strat para comparar sempre os strat_support_assets com os model.asset
                strat_support_assets = strat_data.get('strat_support_assets', {})

                 # Aqui vou fazer a verificação dos timeframes do strat_support_assets dessa Strat
                for supp_asset_name, supp_asset_obj in strat_support_assets.items():
                    timeframes = getattr(supp_asset_obj, "timeframe", [])
                    if not isinstance(timeframes, (list, tuple, set)): timeframes = [timeframes]  
                    for tf in timeframes:
                        if tf and tf != self.operation_timeframe:
                            unique_timeframes_all.setdefault(supp_asset_name, []).append(tf)

                # If any different timeframes found then transfers HTF to LTF, if any tf < operational_timeframe then interrupts operation
                if unique_timeframes_all:
                    print("       ⚠️  Different timeframes found:", unique_timeframes_all)

                    # Creates a operational template df to fit HTF data
                    ltf_asset_obj = next(
                        (
                            asset for asset in model_assets.values()
                            if isinstance(getattr(asset, "timeframe", None), (list, tuple, set))
                            and self.operation_timeframe in getattr(asset, "timeframe")
                            or getattr(asset, "timeframe", "").upper() == self.operation_timeframe.upper()
                        ),
                        None
                    )

                    if ltf_asset_obj is not None:
                        ltf_template_df = ltf_asset_obj.data_get(self.operation_timeframe).copy()
                        if not ltf_template_df.empty:
                            ltf_template_df = create_datetime_columns(ltf_template_df) # Garantees datetime columns
                            ltf_template_df = ltf_template_df.dropna(subset=['datetime']) # Removes possible NaT
                            ltf_template_df = ltf_template_df[['datetime', 'date', 'time']].copy() # Keeps only the necessary
                        else: print("           ⚠️ LTF dataframe is empty!")
                    else: print("           ⚠️ No asset found with timeframe == operation_timeframe. THIS SHOULD NOT HAPPEN")

                    # Takes HTF transfers to LTF and saves df object to cache
                    for unique_tf_asset_name, unique_tf in unique_timeframes_all.items(): 
                        print(f"           > unique_tf_asset_name: {unique_tf_asset_name} with unique_tf: {unique_tf}")
                        
                        htf_asset_obj = model_assets.get(unique_tf_asset_name) or strat_support_assets.get(unique_tf_asset_name)
                        if htf_asset_obj:
                            print(f"              > unique tf(s) found for: {unique_tf_asset_name}")
                        else:
                            print(f"              ⚠️ Asset '{unique_tf_asset_name}' not found in model or strat_support_assets")
                            continue
                        
                        for tf in unique_tf:
                            htf_asset_obj_df = htf_asset_obj.data_get(tf) # htf_asset_obj_df now hold the df object of the HTF asset to transfer to LTF for one timeframe of each asset
                            if htf_asset_obj_df is None or htf_asset_obj_df.empty: 
                                print(f"              ⚠️  Warning: No data found for asset '{unique_tf_asset_name}' with timeframe '{tf}'")
                                continue
                            #print(f"                > [{tf}]\n")

                            ltf_df = copy.deepcopy(ltf_template_df) # LTF Template
                            ltf_df = self._transfer_HTF_Columns(ltf_df, self.operation_timeframe, htf_asset_obj_df, tf) # OBS: Datetime seem to be oriented toward the opening of the bar
                            ltf_df.to_excel(f"C:\\Users\\Patrick\\Desktop\\Operation_{model_name}_{strat_name}_{unique_tf_asset_name}_{tf}_to_{self.operation_timeframe}.xlsx", index=False)
                            print(f"                  > Transferred HTF '{tf}' columns to LTF '{self.operation_timeframe}' for asset '{unique_tf_asset_name}'")

                else:
                    print("       ✅  No different timeframes from operational_timeframe found")

                    # Passar df HTF para LTF
                    # Como você tem uma estrutura de OperationResult com cache e compressão, o ideal é:
                    # Fazer o mapping multitimeframe logo após carregar os dados (pré-processamento).
                    # Salvar os dados sincronizados (por exemplo: asset_obj.data['M5_with_H1']).
                    # Usar esse dataframe já pronto durante todo o backtest/sinal.

                # Gets Indicators
                # for ind_key, ind_obj in indicators.items():
                #     print(f"    > asset(s): {ind_obj.asset} | strat: {strat_name} | indicator: {ind_key}")

                # If unique_timeframes > 1 then transfers HTF columns to LTF for each Strat

                # Gets Signal Rules
                

                # Calculates Signals
                
                


        return None


        
    def _preliminary_backtest(self):
        return None



    def _get_asset_object(self, model_name, asset_name: str): # Gets the real Asset object (not mapped metadata)
        models = self._get_all_models()
        model = models.get(model_name)

        if not model: return None

        if isinstance(model.assets, Asset):
            return model.assets if model.assets.name == asset_name else None
        elif isinstance(model.assets, Asset_Portfolio):
            return model.assets.assets.get(asset_name)
        return None

    def _transfer_HTF_Columns(self, ltf_df: pd.DataFrame, ltf_tf: str, htf_df: pd.DataFrame, htf_tf: str, columns: Optional[List[str]] = None): 
        """
        WARNING: Check data source to ensure that 'datetime' represents the bar's opening time like (MT5), yahoo finance uses closing time!
        Transfers specified columns from a higher timeframe (HTF) DataFrame to a lower timeframe (LTF) DataFrame.
        Ensures that HTF data is only used after its bar has closed.
        """
        
        def get_tf_minutes(tf: str) -> int:
            if tf.startswith('M'): return int(tf[1:])
            elif tf.startswith('H'): return int(tf[1:]) * 60
            elif tf.startswith('D'): return int(tf[1:]) * 1440
            else: raise ValueError(f"Timeframe não suportado: {tf}")
            
        if not columns:
            columns = [col for col in htf_df.columns if col != 'datetime']
            
        ltf_minutes = get_tf_minutes(ltf_tf)
        htf_minutes = get_tf_minutes(htf_tf)
        
        if ltf_minutes >= htf_minutes:
            raise ValueError(f"Timeframe menor ({ltf_tf}) deve ser menor que o maior ({htf_tf})")
            
        # Cria cópias
        ltf_df = ltf_df.copy()
        htf_df = htf_df.copy()
        
        if 'datetime' not in ltf_df.columns or 'datetime' not in htf_df.columns:
            raise ValueError("Ambos os DataFrames precisam ter coluna 'datetime'")
            
        # Converte para datetime
        ltf_df['datetime'] = pd.to_datetime(ltf_df['datetime'])
        htf_df['datetime'] = pd.to_datetime(htf_df['datetime'])
        
        # Ordena por datetime
        ltf_df = ltf_df.sort_values('datetime')
        htf_df = htf_df.sort_values('datetime')
        
        # Para usar merge_asof corretamente, precisamos ajustar os timestamps do HTF
        # Uma barra HTF só fica disponível após seu fechamento
        # O fechamento acontece no início da próxima barra HTF
        htf_available = htf_df.copy()
        
        # Cria uma coluna com o timestamp de quando a barra HTF fica disponível
        # (início da próxima barra HTF)
        htf_available['available_from'] = htf_available['datetime'].shift(-1)
        
        # Remove a última linha (não tem próxima barra para definir disponibilidade)
        htf_available = htf_available.iloc[:-1]
        
        # Filtra colunas
        htf_to_merge = htf_available[['available_from'] + columns].copy()
        
        # Agora fazemos o merge usando 'available_from' como chave
        # Isso garante que só usamos a barra HTF quando ela já está completa
        merged_df = pd.merge_asof(
            ltf_df, 
            htf_to_merge, 
            left_on='datetime',
            right_on='available_from',
            direction='backward',
            suffixes=('', f'_{htf_tf}')
        )
        
        # Remove a coluna auxiliar
        merged_df = merged_df.drop('available_from', axis=1, errors='ignore')
        
        return merged_df

    # IV - Execution


    def collect_strat_indicators(strat_obj): # For backtest
        """
        Receives a Strat object and returns all indicators in a structured dict:
        { asset_name: { indicator_name: { param_set_str: df } } }
        """
        result = {}
        
        for ind_key, ind_obj in strat_obj.get('indicators', {}).items():
            assets_to_use = [ind_obj.asset] if ind_obj.asset != "CURR_ASSET" else strat_obj.get('assets', {}).keys()
            
            for asset_name in assets_to_use:
                asset_obj = strat_obj['assets'].get(asset_name) or strat_obj['strat_support_assets'].get(asset_name)
                if asset_obj is None:
                    continue

                data = asset_obj.data_get(ind_obj.timeframe)
                if data is None or data.empty:
                    continue

                # Calcula todos os sets
                calculated_sets = ind_obj.calculate_all_sets(data)
                
                for full_key, df in calculated_sets.items():
                    # full_key já contém os parâmetros, mas podemos extrair apenas o suffix
                    param_str = full_key.split('.')[-1]
                    
                    result.setdefault(asset_name, {}).setdefault(ind_obj.__class__.__name__, {})[param_str] = df
                    
        return result
    
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


    # V - Pos-Processing
    def _calculate_metrics(self):
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
        print(f"\n>>> Init and Validating Operation <<<")
        self._validate_operation()

        # II - Hierarchical Mapping
        print(f"\n>>> Hierarchical Mapping <<<")
        print(f"    > Mapping Structure")
        self._mapping()
        self._operation_result.print_structure()
        print()

        # III - Data Pre-Processing
        print(f"\n>>> Data Pre-Processing <<<")
        print(f"    > Calculating Indicators")
        self._calculates_indicators()
        print(f"    > Generating Signals")
        self._generate_signals()
        print(f"    > Generating Preliminary Results")
        self._preliminary_backtest() # If simple backtest then stops here?

        # IV - Execution
        print(f"\n>>> Executing Operation <<<")
        if isinstance(self.operation, Backtest): self.operation.run()
        elif isinstance(self.operation, Optimization): self.operation.run()
        elif isinstance(self.operation, Walkforward): self.operation.run()

        # V - Pos-Processing
        print(f"\n>>> Pos-Processing <<<")
        if self.metrics: self._calculate_metrics()

        # VI - Saving   
        print(f"\n>>> Saving Results <<<")
        if self.save: self.save_results()

        # VII - Cleanup
        print(f"\n>>> Cleaning Memory <<<")
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
                                self._indicators_cache[cache_key] = ind.calculate(data, param_value)
        return None


"""
    