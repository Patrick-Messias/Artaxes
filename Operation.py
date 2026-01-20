import pandas as pd, numpy as np, json, sys, uuid, copy, datetime, json
sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend\Indicators')
sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend')

from typing import Union, Dict, List, Optional, Any
from dataclasses import dataclass, field
from Model import ModelParams, Model
from Asset import Asset, AssetParams
from Strat import Strat, StratParams, ExecutionSettings, DataSettings, TimeSettings
from OptimizedOperationResult import OptimizedOperationResult
from Portfolio import Portfolio, PortfolioParams
from Backtest import Backtest, BacktestParams
from ModelMoneyManager import ModelMoneyManager, ModelMoneyManagerParams
from StratMoneyManager import StratMoneyManager, StratMoneyManagerParams
from ModelSystemManager import ModelSystemManager, ModelSystemManagerParams
from Optimization import Optimization
from Walkforward import Walkforward
from Indicator import Indicator
from BaseClass import BaseClass
from Persistance import Persistance
from itertools import product

from Trade import Trade

from MA import MA # type: ignore

# =========================================================================================================================================|| Global Mapping (REMOVE FROM TIHS FILE LATER)

_map = { #Não deve mapear os assets, strat, etc porque toda vez vai ter que iterar sobre eles 
    '{portfolio_name}': {
        'models': {
            '{model_name}': {
                'strats': {
                    '{strat_name}': {
                        'assets': {
                            '{asset_name}': {
                                'param_sets': {
                                    '{param_set_name}': {
                                        'param_set_dict': {
                                            dict # ['portfolio_name']['models'][model_name]['strats']['strat_name']['param_sets']['param_set']['param_set_dict']: dict
                                        },
                                        'signals': {
                                            pd.DataFrame # ['portfolio_name']['models'][model_name]['strats']['strat_name']['param_sets']['param_set']['signals']: pd.DataFrame
                                        },
                                        'trades': {
                                            list[Trade] # ['portfolio_name']['models'][model_name]['strats']['strat_name']['param_sets']['param_set']['preliminary_backtest']: np.array['preliminary_pnl']
                                        }
                                    },
                                    'walkforward': {
                                        '{wf_param}': {
                                            list[Trade] # ['portfolio_name']['models'][model_name]['strats']['strat_name']['walkforward'][wf_param]: Walkforward
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    },
    'indicators': {
        '{ind_calc_name}': {
            '{ind_asset_name}': {
                '{ind_timeframe}': {
                    '{ind_param_set_key}': pd.DataFrame # self._results_map['indicators'][ind_calc_name][ind_asset_name][ind_timeframe][ind_param_set_key]
                }
            }
        } 
    }
}



# =========================================================================================================================================|| Global Mapping

@dataclass
class OperationParams():
    name: str = field(default_factory=lambda: f'model_{uuid.uuid4()}')
    data: Union[Model, Portfolio]=None # Can make an operation with a single model or portfolio
    operation: Union[Backtest, Optimization, Walkforward]=None 
    pre_backtest_signal_is_position: bool=False
    assets: Optional[Dict[str, Any]] = field(default_factory=dict) # Global Assets

    # Metrics
    metrics: Optional[Dict[str, Indicator]] = field(default_factory=dict)

    # Settings
    operation_backtest_all_signals_are_positions: bool=False
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
        self.pre_backtest_signal_is_position = op_params.pre_backtest_signal_is_position
        self.assets = op_params.assets 

        self.metrics = op_params.metrics

        self.operation_backtest_all_signals_are_positions = op_params.operation_backtest_all_signals_are_positions
        self.operation_timeframe = op_params.operation_timeframe
        self.date_start = op_params.date_start
        self.date_end = op_params.date_end
        self.save = op_params.save

        self._results_map = {} #OptimizedOperationResult() # NOTE REDO OptimizedOperationResult() for new structure NOTE
        self.unique_datetime_df = pd.DataFrame

        # Optimized Cache WIP to save data
        #self._memory_cache = {}
        #self._cache_size_limit = 100 * 1024 * 1024  # 100MB limit

    # 1 - Data Pre-Processing
    def _data_pre_processing(self):
        models = self._get_all_models()
        self._results_map[self.name] = {'models': {}}

        for model_name, model_obj in models.items():
            strats = model_obj.strat
            assets = model_obj.assets
            self._results_map[self.name]['models'][model_name] = {'strats': {}}

            for strat_name, strat_obj in strats.items():
                params = strat_obj.params
                strat_indicators = strat_obj.indicators
                self._results_map[self.name]['models'][model_name]['strats'][strat_name] = {'assets': {}}

                # 1 - Calculates Param Sets
                param_sets = self._calculate_param_combinations(params)

                # 2 - Calculates Signals and Indicators
                for asset_name in assets: # dict[str]
                    asset_class = self.assets.get(asset_name, None) # Asset Class
                    if asset_class is None: raise ValueError(f"Asset '{asset_name}' not found in global assets.")
                    self._results_map[self.name]['models'][model_name]['strats'][strat_name]['assets'][asset_name] = {'param_sets': {}}
                    
                    if isinstance(self.operation, Walkforward):
                        isos = self.operation.isos if self.operation.isos is not None else ['12_12']
                        self._results_map[self.name]['models'][model_name]['strats'][strat_name]['assets'][asset_name]['param_sets'] = {'walkforward': {}}
                        self._results_map[self.name]['models'][model_name]['strats'][strat_name]['assets'][asset_name]['param_sets']['walkforward'] = {iso: None for iso in isos}

                    for param_set_name, param_set_dict in param_sets.items():
                        curr_asset_obj = asset_class.data_get(self.operation_timeframe) #self._resolve_asset(asset_name, self.operation_timeframe) #asset_class.data_get(self.operation_timeframe) # Gets current's Asset data
                        signals = strat_obj.signal_rules # Gets signal functions
                        self._results_map[self.name]['models'][model_name]['strats'][strat_name]['assets'][asset_name]['param_sets'][param_set_name] = {'param_set_dict': param_set_dict, 'signals': None, 'trades': None}
                        
                        # Calculates Indicators

                        # NOTE   Corrigir, se asset=None então tem que criar um novo para cada Asset   NOTE

                        for ind_name, ind_obj in strat_indicators.items():
                            if ind_obj is None:
                                print(f'       > {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Indicator {ind_name} is None, skipping calculation.')
                                continue

                            curr_asset_obj = self._calculate_indicator(ind_name, ind_obj, param_set_dict, curr_asset_obj, asset_name, asset_class.datetime_candle_references) # Calculates each indicator and saves in the global mapping

                        for curr_signal_def_name, curr_signal_def_obj in signals.items():
                            if curr_signal_def_obj is not None:
                                curr_asset_obj[curr_signal_def_name] = curr_signal_def_obj(self, asset_name, curr_asset_obj, param_set_dict)

                                num_true_signals = curr_asset_obj[curr_signal_def_name].sum()
                                print(f'       > {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Calculating Signal: {curr_signal_def_name} - Model: {model_name} - Strat: {strat_name} - Asset: {asset_name} - True count: {num_true_signals}/{len(curr_asset_obj)}')
                        #print(curr_asset_obj)   
                        #curr_asset_obj.to_excel(f'C:\\Users\\Patrick\\Desktop\\Model_{model_name}_Strat_{strat_name}_Asset_{asset_name}_ParamSet_{param_set_name}_Signals.xlsx', index=False)
                        self._results_map[self.name]['models'][model_name]['strats'][strat_name]['assets'][asset_name]['param_sets'][param_set_name]['signals'] = curr_asset_obj # Saves full DataFrame with signals and indicators for exporting to C++ backtest later
        return True
    
    # 2 - Serialize Data to JSON for C++ Backtest
    def _serialize_to_json(self):
        import json
        data = {}
        
        # Iterar sobre _results_map para DataFrames de sinais
        if self.name in self._results_map and 'models' in self._results_map[self.name]:
            for model_name, model_data in self._results_map[self.name]['models'].items():
                if 'strats' in model_data:
                    for strat_name, strat_data in model_data['strats'].items():
                        if 'assets' in strat_data:
                            for asset_name, asset_data in strat_data['assets'].items():
                                if 'param_sets' in asset_data:
                                    for param_set_name, param_set_data in asset_data['param_sets'].items():
                                        if 'signals' in param_set_data and isinstance(param_set_data['signals'], pd.DataFrame):
                                            key = f"{model_name}_{strat_name}_{asset_name}_{param_set_name}"
                                            data[key] = param_set_data['signals'].to_json()
        
        # Adicionar variáveis
        data['pre_backtest_signal_is_position'] = self.pre_backtest_signal_is_position
        data['date_start'] = self.date_start
        data['date_end'] = self.date_end
        
        return json.dumps(data)



    # || ===================================================================== || Helper Functions || ===================================================================== ||

    def _calculate_indicator(self, ind_calc_name: str, ind_calc_obj, param_set_dict, curr_asset_df_obj: pd.DataFrame=None, curr_asset_name: str=None, datetime_reference_candles='open'): # Calculates each individual indicator and saves in the global mapping

        ind_timeframe = ind_calc_obj.timeframe
        if ind_calc_obj.asset is None: 
            ind_asset_name = curr_asset_name
        else: ind_asset_name = ind_calc_obj.asset

        # Decomposes param_set to only those relevant to this indicator, to avoid recalculating for unrelated params
        ind_param_set_obj = self.effective_params_from_global(ind_calc_obj.params, param_set_dict)
        ind_param_set_key = self.param_suffix(ind_param_set_obj)
        ind_calc_obj.__dict__['params'] = ind_param_set_obj

        # Check if already calculated
        if ind_calc_name in self._results_map.get('indicators', {}) and \
        ind_asset_name in self._results_map['indicators'][ind_calc_name] and \
        ind_timeframe in self._results_map['indicators'][ind_calc_name][ind_asset_name] and \
        ind_param_set_key in self._results_map['indicators'][ind_calc_name][ind_asset_name][ind_timeframe]:
            ind_column_df = self._results_map['indicators'][ind_calc_name][ind_asset_name][ind_timeframe][ind_param_set_key]
        else:
            # Calculate the indicator
            if ind_asset_name == curr_asset_name or ind_asset_name is None:
                # Same asset or no specific asset (use current)
                df_used = curr_asset_df_obj
                ind_result = ind_calc_obj.calculate(curr_asset_df_obj)
                print(f'       > {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Calculating Indicators - {ind_calc_name} - Ind Asset: {ind_calc_obj.asset} - Idx Asset {curr_asset_name} - Timeframe: {ind_timeframe} - Param Set: {ind_param_set_key}')
            else:
                # Different asset: get the data for the indicator asset
                asset_class = global_assets.get(ind_asset_name, None)
                if asset_class is None:
                    raise ValueError(f"Asset '{ind_asset_name}' not found in global assets.")
                datetime_reference_candles = asset_class.datetime_candle_references
                ind_asset_df = asset_class.data_get(ind_timeframe)
                df_used = ind_asset_df
                ind_result = ind_calc_obj.calculate(ind_asset_df)
                print(f'       > {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Calculating Indicators - {ind_calc_name} - Ind Asset: {ind_calc_obj.asset} - Idx Asset {curr_asset_name} - Timeframe: {ind_timeframe} - Param Set: {ind_param_set_key}')
            
            # Ensure ind_column_df is a DataFrame with proper column name and datetime
            if isinstance(ind_result, pd.Series):
                ind_column_df = pd.DataFrame({'datetime': df_used['datetime'], ind_calc_name: ind_result})
            else:
                ind_column_df = ind_result  # Assume DataFrame if multiple columns

            # Save to mapping
            if 'indicators' not in self._results_map:
                self._results_map['indicators'] = {}
            if ind_calc_name not in self._results_map['indicators']:
                self._results_map['indicators'][ind_calc_name] = {}
            if ind_asset_name not in self._results_map['indicators'][ind_calc_name]:
                self._results_map['indicators'][ind_calc_name][ind_asset_name] = {}
            if ind_timeframe not in self._results_map['indicators'][ind_calc_name][ind_asset_name]:
                self._results_map['indicators'][ind_calc_name][ind_asset_name][ind_timeframe] = {}
            self._results_map['indicators'][ind_calc_name][ind_asset_name][ind_timeframe][ind_param_set_key] = ind_column_df
            
        # Now, add the columns to curr_asset_df_obj
        if ind_timeframe == self.operation_timeframe:
            if ind_asset_name == curr_asset_name or ind_asset_name is None: # Same asset and timeframe: just add columns
                for col in ind_column_df.columns:
                    if col not in curr_asset_df_obj.columns:
                        curr_asset_df_obj[col] = ind_column_df[col]
            else: # Different asset, same timeframe: merge on datetime intersection
                # Align by datetime
                merged = pd.merge(curr_asset_df_obj, ind_column_df, on='datetime', how='left', suffixes=('', f'_{ind_calc_name}'))
                # Rename columns if needed, but for now, add as is
                for col in ind_column_df.columns:
                    if col != 'datetime' and col not in curr_asset_df_obj.columns:
                        curr_asset_df_obj[col] = merged[col]
        else:  # Different timeframe: transfer HTF to LTF
            # Use transfer_htf_columns to transfer ind_column_df (HTF) to curr_asset_df_obj (LTF) / Assume ind_timeframe is HTF, self.operation_timeframe is LTF
            transferred = self.transfer_htf_columns(
                ltf_df=curr_asset_df_obj,
                ltf_tf=self.operation_timeframe,
                htf_df=ind_column_df,
                htf_tf=ind_timeframe,
                datetime_reference_candles=datetime_reference_candles,  # Assuming MT5 style
                columns=[col for col in ind_column_df.columns if col != 'datetime'],
                add_htf_tag=False
            )
            # Add the transferred columns to curr_asset_df_obj
            htf_cols = [col for col in transferred.columns if col.endswith(f"_{ind_timeframe}") and col not in curr_asset_df_obj.columns]
            for col in transferred.columns:
                curr_asset_df_obj[col] = transferred[col]

        return curr_asset_df_obj

    def _get_all_models(self) -> dict: # Returns all Model(s) from data
        if isinstance(self.data, Model): # Single Model
            return {self.data.name: self.data}
        elif isinstance(self.data, Portfolio): # Portfolio
            return self.data.get_all_models()
        else: return {}

    def _resolve_asset(self, asset: str, timeframe: str, curr_asset_df_obj: pd.DataFrame=None, date_start: pd.Timestamp=None, date_end: pd.Timestamp=None, columns: list[str]=None) -> pd.DataFrame: 
        asset_class = self.assets.get(asset, None)
        if asset_class is None:
            raise ValueError(f"Asset '{asset}' not found in global assets.")
        df = asset_class.data_get(timeframe)

        if columns is not None:
            df = df[columns + (['datetime'] if 'datetime' in df.columns else ['date'] if 'date' in df.columns else [])]

        # different timeframes HTF -> LTF
        if timeframe != self.operation_timeframe:
            ltf_df = copy.deepcopy(curr_asset_df_obj) # LTF Template
            df = self.transfer_htf_columns(ltf_df, self.operation_timeframe, df, timeframe, asset_class.datetime_candle_references)

        # if not columns is empty: transfers columns 
        return df
    
    def effective_params_from_global(self, ind_defaults, global_ps):
        eff = {}
        for k, v in ind_defaults.items():
            if isinstance(v, str) and v in global_ps:
                eff[k] = global_ps[v]
            elif k in global_ps:
                eff[k] = global_ps[k]
            else:
                eff[k] = v
        return eff
    
    def param_suffix(self, params: dict, sep: str = "-", pair_sep: str = "") -> str:
        # Gera um sufixo determinístico a partir de `params`.
        # - Ordena chaves para garantir determinismo.
        # - Normaliza listas/tuplas/range/dict para representações consistentes.
        # - Retorna uma string segura para usar como key/cache/lookup.
     
        def _norm(v):
            if v is None:
                return "None"
            if isinstance(v, range):
                return f"range({v.start},{v.stop},{v.step})"
            if isinstance(v, (list, tuple)):
                return "[" + ",".join(str(x) for x in v) + "]"
            if isinstance(v, dict):
                return json.dumps(v, sort_keys=True, separators=(",", ":"))
            # Fallback: booleans, numbers, strings, objects
            return str(v)

        parts = []
        for k in sorted(params.keys()):
            v = params[k]
            parts.append(f"{k}{sep}{_norm(v)}")
        return "_".join(parts)
    
    def _calculate_param_combinations(self, param_dict, prefix="param_set"): # Recebe um dict de parâmetros e gera todas as combinações possíveis. Retorna um dict estruturado.
        # Separa parâmetros que variam e parâmetros fixos
        varying = {}
        fixed = {}
        
        for key, val in param_dict.items(): # Considera 'valor único' se não for iterável útil (range, list, tuple)
            if isinstance(val, (list, tuple, range)):
                varying[key] = list(val)
            else:
                fixed[key] = val

        # Se não houver parâmetros variados, apenas retorna o original
        if not varying:
            name = f"{prefix}_{'-'.join(str(v) for v in fixed.values())}"
            return {name: param_dict}

        # Gera combinações
        keys = list(varying.keys())
        values = [varying[k] for k in keys]

        result = {}

        for combo in product(*values):
            combo_dict = dict(zip(keys, combo)) | fixed # monta dict final
            combo_name = f"{prefix}-" + "-".join(str(combo_dict[k]) for k in combo_dict) # cria nome único
            result[combo_name] = combo_dict # add

        return result

    def transfer_htf_columns(
        self,
        ltf_df: pd.DataFrame,
        ltf_tf: str,
        htf_df: pd.DataFrame,
        htf_tf: str,
        datetime_reference_candles: str = 'open',  # 'open' (MT5) ou 'close'
        columns: Optional[List[str]] = None,
        add_htf_tag: bool = True
    ) -> pd.DataFrame:
        # Transfers HTF columns to LTF dataframe without lookahead bias.
        # HTF values are only available AFTER the HTF candle has closed.
        # Assumptions: - datetime column represents candle OPEN time (MT5-like) or CLOSE time.
        
        def get_tf_minutes(tf: str) -> Optional[int]:
            if tf.startswith('M') and not tf.startswith('MN'):
                return int(tf[1:])
            elif tf.startswith('H'):
                return int(tf[1:]) * 60
            elif tf.startswith('D'):
                return int(tf[1:]) * 1440
            else:
                return None  # W, MN → handled by timestamps, not minutes

        # Defensive copies
        ltf_df = ltf_df.copy()
        htf_df = htf_df.copy()

        # Datetime normalization
        ltf_df['datetime'] = pd.to_datetime(ltf_df['datetime'])
        htf_df['datetime'] = pd.to_datetime(htf_df['datetime'])

        ltf_df = ltf_df.sort_values('datetime')
        htf_df = htf_df.sort_values('datetime')

        if columns is None:
            columns = list(htf_df.columns)
            columns.remove('datetime')

        # Determine candle duration
        htf_minutes = get_tf_minutes(htf_tf)

        # Shift HTF timestamps so they become available ONLY after close
        htf_aligned = htf_df[['datetime'] + columns].copy()

        if datetime_reference_candles == 'open':
            if htf_minutes is not None:
                htf_aligned['datetime'] += pd.Timedelta(minutes=htf_minutes)
            # W / MN → already safe using timestamps only

        elif datetime_reference_candles != 'close':
            raise ValueError("datetime_reference_candles must be 'open' or 'close'")

        # Rename HTF columns, avoiding duplication
        renamed_columns = {}
        for col in columns:
            if add_htf_tag is False or col.endswith(f"_{htf_tf}"):
                renamed_columns[col] = col  # Keep as is
            else:
                renamed_columns[col] = f"{col}_{htf_tf}"
        htf_aligned = htf_aligned.rename(columns=renamed_columns)

        # Drop overlapping columns from ltf_df to avoid merge conflicts
        overlapping_cols = set(ltf_df.columns) & set(htf_aligned.columns) - {'datetime'}
        ltf_df = ltf_df.drop(columns=overlapping_cols)

        # ASOF merge (forward to replicate HTF value to LTF of the same day)
        merged = pd.merge_asof(
            ltf_df,
            htf_aligned,
            on='datetime',
            direction='forward'  # Changed to forward for correct replication
        )

        # Remove HTF values before first HTF close
        first_valid_time = htf_aligned['datetime'].iloc[0]
        if add_htf_tag is False:
            htf_cols = [renamed_columns[col] for col in columns]
        else:
            htf_cols = [col for col in merged.columns if col.endswith(f"_{htf_tf}")]
        merged.loc[merged['datetime'] < first_valid_time, htf_cols] = np.nan
    
        return merged



    # # 2. Run Operation
    # def _run_operation(self):
    #     models = self._get_all_models()

    #     if isinstance(self.operation, Walkforward): 
    #         wfm_sets = self.operation.calculate_walkforward_matrix_sets()


    #     for model_name, model_obj in models.items():
    #         strats = model_obj.strat
    #         assets = model_obj.assets


    #          IMPORTANTE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!s
    #         Não fazer case1/case2, apenas gerar todos os trades, salvnado informações necessárias para a operação, depois analisar wf/portfolio,etc
    #         # Case 1 -> Backtest de Portfolio itera sobre unique datetime e cria posição de Strat-Asset-ParamSet(s) baseado em regras do modelo
    #         # Case 2 -> Backtest onde cada Strat tem todo o seu próprio conjunto de assets e param_sets

    #         for strat_name, strat_obj in strats.items():
    #             params = strat_obj.params
    #             strat_indicators = strat_obj.indicators


        #C++ CODE
        # NOTE Enside C++ Backtest code must receive wf_matrix and iterate and save each result from each wfm
        # for mtx_key, mtx_dict in wf_matrix.items(): # WF IS-OS 12-12, 12-6, 12-3, etc
        # ✔️ Python prepara tudo → monta _results_map, resolve assets, indicadores, sinais, param_sets, HTF→LTF, etc.
        # ✔️ C++ só recebe blocos prontos → apenas um DataFrame final por asset + signals + param_set e executa somente a lógica de backtest /
        # (loop, posição, PnL, SL/TP, etc.).
                            

    def run(self):

        # I - Init and Validation
        print(f"\n>>> Init and Validating Operation <<<")
    #    self._validate_operation()

        # II - Data Pre-Processing
        print(f"\n>>> Data Pre-Processing - Calculating Param Sets, Indicators and Signals <<<")
        self._data_pre_processing()

        # Não se faz preliminary backtest, vai direto para backtest(s) em C++

        # ->>>>>>>> ORDERM ABAXIO
        # 1. Testar até aqui, verificar que está tudo fucnionando
        # 2. Planejar bem e criar o código em C++ para Backtest, Walkforward, WFM, etc
        # 3. Retornar para python com os resultados dos backtests para analise de Portfolio
        # 4. NOTE If Walkforward then at each IS saves data while Backtest then compares results, so even if not Day Trade still doesn't have bias  /
        # Operation checks if Walkforward then uses class to save data at each IS, then gets OS results from backtests

        # III - Data Processing
        print(f"\n>>> Data Processing - Serializing Data for C++ <<<")
        self._serialize_to_json()

        # VI - Execution
        print(f'\n>>> Executing Operation {type(self.operation).__name__} in C++ <<<')

        # V - Pos-Processing
        print(f"\n>>> Pos-Processing <<<")
        if self.metrics: self._data_pos_processing()

        # VI - Saving   
        print(f"\n>>> Saving Results <<<")
        if self.save: self.save_results()

        # VII - Cleanup
        print(f"\n>>> Cleaning Memory <<<")
        self.cleanup_memory()

        return self._operation_result



if __name__ == "__main__":

    eurusd = Asset(
        name='EURUSD',
        type='currency_pair',
        market='forex',
        data_path=f'C:\\Users\\Patrick\\Desktop\\Artaxes Portfolio\\MAIN\\MT5_Dados\\Forex'
    )
    gbpusd = Asset(
        name='GBPUSD',
        type='currency_pair',
        market='forex',
        data_path=f'C:\\Users\\Patrick\\Desktop\\Artaxes Portfolio\\MAIN\\MT5_Dados\\Forex'
    )
    usdjpy = Asset(
        name='USDJPY',
        type='currency_pair',
        market='forex',
        data_path=f'C:\\Users\\Patrick\\Desktop\\Artaxes Portfolio\\MAIN\\MT5_Dados\\Forex'
    )

    global_assets = {'EURUSD': eurusd, 'GBPUSD': gbpusd, 'USDJPY': usdjpy} # Global Assets, loaded when app starts up, has all Asset and Portfolios 


    model_assets=['GBPUSD', 'EURUSD', 'USDJPY'] # Only keys #, 'GBPUSD'
    model_execution_tf = 'M15'

    Params = {
        'AT15': { 
            'execution_tf': model_execution_tf,
            'sl_perc': range(2, 2+1, 1), # 3
            'param1': range(20, 20+1, 30), #50
            'param2': range(2, 2+1, 1), # 3
            'param3': ['sma'] #, 'ema'
        }
    }

    # User imput Indicators
    ind = { 
        'sma': MA(asset=None, timeframe=model_execution_tf, window='param1', type='param3', price_col='close'),
        'ema_htf': MA(asset='GBPUSD', timeframe='D1', window=252, type='ema') #, price_col='open'
    }

    def entry_long(self, curr_asset_name: str, df: pd.DataFrame, curr_param_set: dict): 
        df_D1 = self._resolve_asset(curr_asset_name, 'D1', df)
        df_EURUSD_D1 = self._resolve_asset('EURUSD', 'D1', df)

        sl_perc = curr_param_set['sl_perc']
        diff = df['close']*(sl_perc/100)
        ema_htf = df['ema_htf']

        signal = ((df['close'] > df['sma']) )#& df['close'].shift(1) < df['sma'].shift(1)) # (df['close'] > df['sma'] + diff) & (df['close'].shift(1) < df['sma'].shift(1) + diff) & (df_D1['close'] > df_D1['close'].shift(1))
        return signal
    
    def entry_short(self, curr_asset_name: str, df: pd.DataFrame, curr_param_set: dict): 
        df_D1 = self._resolve_asset(curr_asset_name, 'D1', df)
        df_EURUSD_D1 = self._resolve_asset('EURUSD', 'D1', df)

        sl_perc = curr_param_set['sl_perc']
        diff = df['close']*(sl_perc/100)
        ema_htf = df['ema_htf']

        signal = ((df['close'] < df['sma']) )#& df['close'].shift(1) > df['sma'].shift(1)) # (df['close'] < df['sma'] - diff) & (df['close'].shift(1) > df['sma'].shift(1) - diff) & (df_D1['close'] < df_D1['close'].shift(1))
        return signal

    def exit_tf_long(self, curr_asset_name: str, df: pd.DataFrame, curr_param_set: dict):
        return (df['close'] < df['sma']) 
    def exit_tf_short(self, curr_asset_name: str, df: pd.DataFrame, curr_param_set: dict):
        return (df['close'] > df['sma'])

    AT15 = Strat(
        StratParams(
            name="AT15",
            execution_settings=ExecutionSettings(order_type='market', offset=0.0),
            data_settings=DataSettings(fill_method='ffill', fillna=0),
            mma_settings=None, # If mma_rules=None then will use default or PMA or other saved MMA define in Operation. Else it creates a temporary MMA with mma_settings
            params=Params['AT15'], # SE signal_params então iterar apenas nos parametros do signal_params para criar sets, else usa apenas sets do indicadores, else sem sets
            time_settings=TimeSettings(day_trade=False, timeTI=None, timeEF=None, timeTF=None, next_index_day_close=False, friday_close=False, timeExcludeHours=None, dateExcludeTradingDays=None, dateExcludeMonths=None),
            indicators=ind,
            signal_rules={
                'entry_long': entry_long,
                'entry_short': entry_short,
                'exit_tf_long': exit_tf_long,
                'exit_tf_short': exit_tf_short,
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
            }
        )
    )

    model_1 = Model(
        ModelParams(
            name='MA Trend Following',
            assets=model_assets, # CURR_ASSET refers to this one in strat_support_assets
            strat={'AT15': AT15},
            execution_timeframe=model_execution_tf,
            model_money_manager=ModelMoneyManager(ModelMoneyManagerParams(name="Model1_MM")),
            model_system_manager=None  # Optional - will use default system management
        )
    )



    operation = Operation(
        OperationParams(
            name='operation_test',
            data=model_1,
            operation=Backtest(BacktestParams(name='backtest_test')),
            pre_backtest_signal_is_position=False,
            operation_backtest_all_signals_are_positions=False,
            assets=global_assets,
            operation_timeframe=model_execution_tf,
            date_start='2020-01-01',
            date_end='2023-01-01',
            save=False,
            metrics={}
        )
    )

    operation.run()














"""

    def transfer_htf_columns(
        self,
        ltf_df: pd.DataFrame,
        ltf_tf: str,
        htf_df: pd.DataFrame,
        htf_tf: str,
        datetime_reference_candles: str = 'open',  # 'open' (MT5) ou 'close'
        columns: Optional[List[str]] = None,
        add_htf_tag: bool = True
    ) -> pd.DataFrame:
        # Transfers HTF columns to LTF dataframe without lookahead bias.
        # HTF values are only available AFTER the HTF candle has closed.
        # Assumptions: - datetime column represents candle OPEN time (MT5-like) or CLOSE time.
        
        def get_tf_minutes(tf: str) -> Optional[int]:
            if tf.startswith('M') and not tf.startswith('MN'):
                return int(tf[1:])
            elif tf.startswith('H'):
                return int(tf[1:]) * 60
            elif tf.startswith('D'):
                return int(tf[1:]) * 1440
            else:
                return None  # W, MN → handled by timestamps, not minutes

        # Defensive copies
        ltf_df = ltf_df.copy()
        htf_df = htf_df.copy()

        # Datetime normalization
        ltf_df['datetime'] = pd.to_datetime(ltf_df['datetime'])
        htf_df['datetime'] = pd.to_datetime(htf_df['datetime'])

        ltf_df = ltf_df.sort_values('datetime')
        htf_df = htf_df.sort_values('datetime')

        if columns is None:
            columns = list(htf_df.columns)
            columns.remove('datetime')

        # Determine candle duration
        htf_minutes = get_tf_minutes(htf_tf)

        # Shift HTF timestamps so they become available ONLY after close
        htf_aligned = htf_df[['datetime'] + columns].copy()

        if datetime_reference_candles == 'open':
            if htf_minutes is not None:
                htf_aligned['datetime'] += pd.Timedelta(minutes=htf_minutes)
            # W / MN → already safe using timestamps only

        elif datetime_reference_candles != 'close':
            raise ValueError("datetime_reference_candles must be 'open' or 'close'")

        # Rename HTF columns, avoiding duplication
        renamed_columns = {}
        for col in columns:
            if add_htf_tag is False or col.endswith(f"_{htf_tf}"):
                renamed_columns[col] = col  # Keep as is
            else:
                renamed_columns[col] = f"{col}_{htf_tf}"
        htf_aligned = htf_aligned.rename(columns=renamed_columns)

        # ASOF merge (forward to replicate HTF value to LTF of the same day)
        merged = pd.merge_asof(
            ltf_df,
            htf_aligned,
            on='datetime',
            direction='forward'  # Changed to forward for correct replication
        )

        # Remove HTF values before first HTF close
        first_valid_time = htf_aligned['datetime'].iloc[0]
        if add_htf_tag is False:
            htf_cols = [renamed_columns[col] for col in columns]
        else:
            htf_cols = [col for col in merged.columns if col.endswith(f"_{htf_tf}")]
        merged.loc[merged['datetime'] < first_valid_time, htf_cols] = np.nan
   
        return merged
    
    

    def _resolve_indicator(self, all_indicators_from_strat: dict, param_set: dict, all_assets_global, curr_asset_df_obj: pd.DataFrame=None, curr_asset_name: str=None): #ind_name: str, 

        ind_obj = all_indicators_from_strat[ind_name]
        ind_timeframe = ind_obj.timeframe
        ind_asset_name = ind_obj.asset

        # Decomposes param_set to only those relevant to this indicator, to avoid recalculating for unrelated params
        ind_param_set_obj = self.effective_params_from_global(ind_obj.params, param_set)
        ind_param_set_key = self.param_suffix(ind_param_set_obj)

        try: 
            return self._results_map['indicators'][ind_name][ind_asset_name][ind_timeframe][ind_param_set_key]
        except KeyError:
            if curr_asset_df_obj is None or ind_asset_name != curr_asset_name:
                asset_class = all_assets_global.get(ind_asset_name, None)
                if asset_class is None:
                    raise ValueError(f"Asset '{ind_asset_name}' not found in global assets.")
                ind_asset_df_obj = asset_class.data_get(ind_asset_name, ind_timeframe)
            self.ind_obj.params = ind_param_set_obj # Updates indicator params with current set
            ind_column_df = ind_obj.calculate(ind_asset_df_obj) 

        # different timeframes HTF -> LTF
        if ind_timeframe != self.operation_timeframe:
            ltf_df = copy.deepcopy(curr_asset_df_obj) # LTF Template
            if 'datetime' in ind_asset_df_obj.columns:
                ind_column_df['datetime'] = ind_asset_df_obj['datetime']
            elif 'date' in ind_asset_df_obj.columns:
                ind_column_df['date'] = ind_asset_df_obj['date']
            ltf_df = [['datetime']] if 'ind_column_df' in ind_asset_df_obj.columns else [['date']]
            ind_column_df = self._transfer_HTF_Columns(ltf_df, self.operation_timeframe, ind_column_df, ind_asset_df_obj.timeframe, ind_asset_df_obj.datetime_candle_references)

        return ind_column_df 



"""