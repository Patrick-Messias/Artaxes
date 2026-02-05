import polars as pl, numpy as np, json, sys, uuid, copy, datetime, psutil
sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend\Indicators')
sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend')

from typing import Union, Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict, is_dataclass
from Model import ModelParams, Model
from Asset import Asset, AssetParams
from Strat import Strat, StratParams, ExecutionSettings, DataSettings, TimeSettings
from Portfolio import Portfolio, PortfolioParams
from Backtest import Backtest, BacktestParams
from ModelMoneyManager import ModelMoneyManager, ModelMoneyManagerParams
from StratMoneyManager import StratMoneyManager, StratMoneyManagerParams
from ModelSystemManager import ModelSystemManager, ModelSystemManagerParams
from Optimization import Optimization
from Walkforward import Walkforward
from Indicator import Indicator
from BaseClass import BaseClass
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
                                        # 'signals': {
                                        #     pd.DataFrame # ['portfolio_name']['models'][model_name]['strats']['strat_name']['param_sets']['param_set']['signals']: pd.DataFrame
                                        # },
                                        'trades': {
                                            list[Trade] # ['portfolio_name']['models'][model_name]['strats']['strat_name']['param_sets']['param_set']['preliminary_backtest']: np.array['preliminary_pnl']
                                        }
                                    },
                                    'walkforward': {
                                        'wf_param_set':{ # ex: '12_12'
                                            '{wf_param}': {
                                                list[Trade] # ['portfolio_name']['models'][model_name]['strats'][strat_name]['assets'][asset_name]['param_sets']['walkforward'][wf_param_set][wf_param]: Walkforward
                                            }
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
    '{asset_name}': { # Indicators
        '{cache_key}': pl.Series # self._results_map[asset_name][cache_key]
    }
}

# =========================================================================================================================================|| Global Mapping

@dataclass
class OperationParams():
    name: str = field(default_factory=lambda: f'model_{uuid.uuid4()}')
    data: Union[Model, list[Model]]=None # Can make an operation with a single model or portfolio
    #operation: Union[Backtest, Optimization, Walkforward]=None 
    assets: Optional[Dict[str, Any]] = field(default_factory=dict) # Global Assets

    # Metrics
    metrics: Optional[Dict[str, Indicator]] = field(default_factory=dict)

    # Settings
    operation_backtest_all_signals_are_positions: bool=True # If True, all signals are treated as position signals (entry/exit), else treated as simple signals (entry only, exit by SL/TP/Time)
    operation_timeframe: str=None
    date_start: str=None
    date_end: str=None
    save: bool=False
    
class Operation(BaseClass):
    def __init__(self, op_params: OperationParams):
        super().__init__()
        self.name = op_params.name
        self.data = op_params.data
        self.assets = op_params.assets 

        self.metrics = op_params.metrics

        self.operation_backtest_all_signals_are_positions = op_params.operation_backtest_all_signals_are_positions
        self.operation_timeframe = op_params.operation_timeframe
        self.date_start = op_params.date_start
        self.date_end = op_params.date_end
        self.save = op_params.save

        self._results_map = {}  
        self.unique_datetime_df = pl.DataFrame

        self._curr_asset: Optional[str] = None
        self._curr_df_context: Optional[pl.DataFrame] = None
        self._curr_tf_context: Optional[str] = None
        self._curr_datetime_references: Optional[str] = None



    # 1. Test HTF->LTF, create excel with temporary datetime added so can test if ok
    # 2. Test batch system
    # 3. Develop backtest system in cpp, test all exits  



    # || ===================================================================== || I - Operation Validation || ===================================================================== ||

    def _validate_operation(self):
        pass

    # || ===================================================================== || II - Data Processing || ===================================================================== ||

    def _operation(self):
        models = self._get_all_models()
        self._results_map[self.name] = {'models': {}}

        avg_param_set_size_mb = None
        batch_payload = {}
        batch_count = 0
        optimal_batch_size = None

        for model_name, model_obj in models.items():
            strats = model_obj.strat
            assets = model_obj.assets
            self._results_map[self.name]['models'][model_name] = {'strats': {}}
            model_tf = model_obj.execution_timeframe
            self._curr_tf_context = model_tf

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
                    self._curr_datetime_references = asset_class.datetime_candle_references

                    if isinstance(strat_obj.operation, Walkforward):
                        isos = strat_obj.operation.isos if strat_obj.operation.isos is not None else ['12_12']
                        self._results_map[self.name]['models'][model_name]['strats'][strat_name]['assets'][asset_name]['param_sets'] = {'walkforward': {}}
                        self._results_map[self.name]['models'][model_name]['strats'][strat_name]['assets'][asset_name]['param_sets']['walkforward'] = {iso: None for iso in isos}

                    # Calculates Indicators
                    # Identifies if indicators are static or dynamic
                    static_inds = {}
                    dynamic_inds = {}
                    for ind_name, ind_obj in strat_indicators.items():
                        if ind_obj is None: continue
                        is_dynamic = any(isinstance(v, str) and v.startswith('param') for v in ind_obj.params.values())
                        if is_dynamic:
                            dynamic_inds[ind_name] = ind_obj
                        else:
                            static_inds[ind_name] = ind_obj

                    # Gets Asset Data
                    base_asset_df = asset_class.data_get(model_tf)
                    self._curr_asset = asset_name
                    
                    # Calculates Static Indicators (unce per Asset)
                    for ind_name, ind_obj in static_inds.items():
                        base_asset_df = self._calculate_indicator(model_tf, ind_name, ind_obj, {}, base_asset_df, asset_name, asset_class.datetime_candle_references)

                    for param_set_name, param_set_dict in param_sets.items():
                        curr_asset_obj = base_asset_df.clone()
                        self._results_map[self.name]['models'][model_name]['strats'][strat_name]['assets'][asset_name]['param_sets'][param_set_name] = {'param_set_dict': param_set_dict, 'trades': None}
                        self._curr_df_context = curr_asset_obj

                        # Calculates Dynamic Indicators
                        for ind_name, ind_obj in dynamic_inds.items():
                            curr_asset_obj = self._calculate_indicator(model_tf, ind_name, ind_obj, param_set_dict, curr_asset_obj, asset_name, asset_class.datetime_candle_references)
                        
                        # Calculates Signals
                        param_set_dict['execution_timeframe'] = model_tf
                        signals = strat_obj.signal_rules # Gets signal functions
                        for curr_signal_def_name, curr_signal_def_obj in signals.items():
                            if curr_signal_def_obj is not None:
                                #curr_asset_obj[curr_signal_def_name] = curr_signal_def_obj(self, curr_asset_obj, param_set_dict)
                                signal_series = curr_signal_def_obj(self, curr_asset_obj, param_set_dict)
                                curr_asset_obj = curr_asset_obj.with_columns([
                                    pl.Series(curr_signal_def_name, signal_series)
                                ])
                                
                                num_true_signals = curr_asset_obj.select(pl.col(curr_signal_def_name).sum()).item() #num_true_signals = curr_asset_obj[curr_signal_def_name].sum()
                                print(f'> {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Calculating Signal: {curr_signal_def_name} - Model: {model_name} - Strat: {strat_name} - Asset: {asset_name} - True count: {num_true_signals}/{len(curr_asset_obj)}')

                        #curr_asset_obj.write_excel(f'C:\\Users\\Patrick\\Desktop\\Model_{model_name}_Strat_{strat_name}_Asset_{asset_name}_ParamSet_{param_set_name}_Signals.xlsx')

                        # Estimates average size for Batch and Payload
                        if avg_param_set_size_mb is None:
                            avg_param_set_size_mb = self._estimate_paramset_size_mb(curr_asset_obj)
                            optimal_batch_size = self._calculate_optimal_batch_size(avg_param_set_size_mb)
                            print(f'> {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Estimated average param set size: {avg_param_set_size_mb:.2f} MB - Optimal batch size: {optimal_batch_size}')

                        # Payload creation for cpp
                        key = f"{model_name}_{strat_name}_{asset_name}_{param_set_name}"
                        batch_payload[key] = {
                            "data": curr_asset_obj,
                            "meta": {
                                "params": param_set_dict,
                                "time_settings": strat_obj.time_settings,
                                "execution_settings": strat_obj.execution_settings
                            }
                        }
                        batch_count += 1

                        if batch_count >= optimal_batch_size:
                            print(f'> {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Processing batch of size: {len(batch_payload)}')
                            trades = self._run_cpp_operation(batch_payload)
                            self._save_trades(trades)
                            batch_payload.clear()
                            batch_count = 0

        if batch_payload:
            print(f'> {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Processing final batch of size: {len(batch_payload)}')
            trades = self._run_cpp_operation(batch_payload)
            self._save_trades(trades)
            batch_payload.clear()

        return True
    
    # || ===================================================================== || III - Portfolio Simulator || ===================================================================== ||

    def _simulate_portfolio(self, portfolio_backtests_dict ='All'):
        # 1. Selects over all models -> strats -> assets -> param_sets, while iterating verifies if any walkforward operation is present
        # 2. Either selects 1 specific param_set for each strat/asset, iterates over all param_sets or select wf_param_set and iterates over walkforward param_sets
        # 3. Selects all unique datetime from selected backtest trades
        # 4. Iterates over datetimes, ranks param_sets based on previous trades results with some metric (ex: equity, profit factor, etc)
        # 5. For each datetime checks for entries and exits on each strat/asset/param_set simulating a portfolio with real money management and trade management rules



        if portfolio_backtests_dict == 'All': # Uses all backtests from _results_map, else if dict with paths, uses only those backtests 
            pass

        pass

    def _walkforward(self):
        # 1. Iterates over all models -> strats -> assets -> param_sets
        # 2. For each param_set, splits data in multiple isos (in-sample and out-of-sample periods)
        # 3. For each iso, selects results from already calculated trade results from backtest operation
        # 4. Analyzes each iso results and aggregates to final walkforward results

        pass

    def _operation_analysis(self):
        pass

    # || ===================================================================== || Execution Functions || ===================================================================== ||

    def _run_cpp_operation(self, batch_payload: dict):
        try:
            # 1. Garante o caminho da DLL
            path_to_dll = r"C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend\build\Release"
            if path_to_dll not in sys.path: 
                sys.path.append(path_to_dll)
            import engine_cpp # type: ignore

            payload_to_send = {"datasets": {}, "meta": {}}
            last_key = None

            for key, content in batch_payload.items():
                last_key = key
                df = content['data'].clone()
                
                # --- TRATAMENTO DE DATETIME (CORREÇÃO DO ERRO) ---
                if 'datetime' in df.columns and df.schema['datetime'] != pl.Utf8:
                    df = df.with_columns(
                        # No Polars moderno, usa-se to_string em vez de format
                        pl.col('datetime').dt.to_string('%Y-%m-%d %H:%M:%S')
                    )

                # --- TRATAMENTO DE SINAIS (PREVENÇÃO DE ERROS DE DTYPE) ---
                signals = [
                    'entry_long', 'entry_short', 
                    'exit_tf_long', 'exit_tf_short', 
                    'exit_tp_long', 'exit_tp_short', 
                    'exit_sl_long', 'exit_sl_short'
                ]
                cols_signals = [c for c in signals if c in df.columns]
                
                # if cols_signals:
                #     # Cast para Int32 antes do fill_null evita ambiguidade com Boolean
                #     df = df.with_columns([
                #         pl.col(c).cast(pl.Int32, strict=False).fill_null(0) for c in cols_signals
                #     ])
                if cols_signals: #  Cast para Int32 antes do fill_null evita ambiguidade com Boolean
                    df = df.with_columns([
                        pl.col(c).cast(pl.Int32).fill_null(0) for c in cols_signals
                    ])
                    # Verificação rápida
                    for c in cols_signals:
                        if df.select(pl.col(c).sum()).item() > 0:
                            print(f"DEBUG: Sinal {c} contém ativações enviadas ao C++.")
                
                # --- TRATAMENTO NUMÉRICO (OHLC e Indicadores) ---
                cols_num = [
                    name for name, dtype in df.schema.items() 
                    if dtype.is_numeric() and name not in signals
                ]
                
                if cols_num:
                    df = df.with_columns([
                        pl.col(c).cast(pl.Float64).fill_null(0.0) for c in cols_num
                    ])

                # Conversão de objetos complexos para dicionários simples
                def to_plain_dict(obj):
                    if is_dataclass(obj): return asdict(obj)
                    if hasattr(obj, 'to_dict'): return obj.to_dict()
                    if isinstance(obj, dict): return {k: to_plain_dict(v) for k, v in obj.items()}
                    if isinstance(obj, list): return [to_plain_dict(x) for x in obj]
                    return obj

                clean_time_settings = to_plain_dict(content.get('time_settings', {}))
                raw_meta = to_plain_dict(content.get('meta', {}))
                
                # Parâmetros numéricos para o C++
                params = raw_meta.get('params', {}) if isinstance(raw_meta, dict) else {}
                clean_params = {}
                for pk, pv in params.items():
                    if pv is None: clean_params[pk] = 0.0
                    elif isinstance(pv, (int, float, np.number)): clean_params[pk] = float(pv)
                    else: clean_params[pk] = str(pv) if pv else ""

                payload_to_send["datasets"][key] = {
                    "data": df.to_dict(as_series=False),
                    "time_settings": clean_time_settings,
                    "meta": {"params": clean_params}
                }

            if last_key:
                payload_to_send["meta"] = payload_to_send["datasets"][last_key]["meta"]

            # Serialização JSON com suporte a tipos de data remanescentes
            def json_serial(obj):
                if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
                    return obj.isoformat()
                return str(obj)

            json_str = json.dumps(payload_to_send, default=json_serial)
            
            print(f'> {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - JSON Ready, calling C++...')
            raw_output = engine_cpp.run(json_str)
            
            return json.loads(raw_output) if isinstance(raw_output, str) else (raw_output or [])

        except Exception as e:
            print(f'< Error in Python-C++ Bridge: {e}')
            import traceback
            traceback.print_exc()
            return []

    def _serialize_batch_to_json(self, batch_payload):
        data = {
            "meta": {
                "date_start": str(self.date_start) if self.date_start else None,
                "date_end": str(self.date_end) if self.date_end else None
            },
            "datasets": {}
        }

        # Função de suporte para tipos NumPy e Datetimes do Polars
        def _json_default(obj):
            # Polars utiliza datetime nativo do Python ou objetos específicos que str() resolve
            if isinstance(obj, (datetime.datetime, datetime.date)):
                return obj.strftime('%Y-%m-%d %H:%M:%S')
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        for key, payload in batch_payload.items():
            # No Polars, to_dict(as_series=False) gera o formato {coluna: [lista_de_valores]}
            # que é o equivalente ao orient='list' do Pandas.
            df_dict = payload["data"].to_dict(as_series=False)

            data["datasets"][key] = {
                "data": df_dict,
                "params": payload["params"],
                "time_settings": asdict(payload["time_settings"]) if is_dataclass(payload["time_settings"]) else payload["time_settings"],
                "execution_settings": asdict(payload["execution_settings"]) if is_dataclass(payload["execution_settings"]) else payload["execution_settings"],
                "signal_rules": payload["signal_rules"]
            }

        # O Polars é muito rigoroso com tipos; a conversão para dict acima já prepara o terreno
        return json.dumps(data, default=_json_default)

    def _save_trades(self, trades):
        # Verifica se 'trades' é de fato uma lista de dicionários
        if not trades or not isinstance(trades, list):
            print("< Erro: 'trades' não é uma lista válida.")
            return

        if len(trades) > 0 and not isinstance(trades[0], dict):
            print(f"< Erro: O primeiro elemento é {type(trades[0])}, esperava-se dicionário.")
            return
            
        print(f"> Gravando {len(trades)} trades no mapa de resultados...")
        
        for trade in trades:
            # O motor C++ retorna o 'asset' contendo a chave composta que enviamos
            full_key = trade.get("asset", "")
            # Ex: 'MA Trend Following_AT15_EURUSD_param_set-2-20-2-sma-M15'
            
            if "param_set" in full_key:
                # Localiza onde terminam as chaves fixas e começam os parâmetros
                idx = full_key.find("param_set")
                header = full_key[:idx-1] 
                param_set_name = full_key[idx:]
                
                parts = header.split('_')
                if len(parts) >= 3:
                    # parts[0] = 'MA Trend Following', parts[1] = 'AT15', parts[2] = 'EURUSD'
                    model_name, strat_name, asset_name = parts[0], parts[1], parts[2]
                    
                    try:
                        # Navega na estrutura do results_map (Dicionário padrão Python)
                        target = self._results_map[self.name]["models"][model_name]["strats"][strat_name]["assets"][asset_name]["param_sets"][param_set_name]
                        
                        if target.get("trades") is None:
                            target["trades"] = []
                        
                        target["trades"].append(trade)
                    except KeyError:
                        # Se a chave não existir (ex: filtro de data ou erro no payload), ignora
                        continue
        
        print(f"> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Trades gravados com sucesso.")

    def _deserialize_from_json(self, json_data):
        # Se o engine_cpp.run já retornar um objeto Python (lista de dicts), não precisa de json.loads
        if isinstance(json_data, str):
            return json.loads(json_data)
        return json_data # Pybind11 geralmente já converte std::vector<Trade> para list[dict]

    def _estimate_paramset_size_mb(self, df: pl.DataFrame):
        return df.estimated_size() / (1024 ** 2) # No Polars, estimated_size() retorna o tamanho em bytes
    
    def _get_available_memory_mb(self):
        return psutil.virtual_memory().available / (1024 ** 2)
    
    def _calculate_optimal_batch_size(self, avg_paramset_size_mb, safety_margin=0.25, max_batch=1000, min_batch=1):
        available_ram_mb = self._get_available_memory_mb()
        usable_ram_mb = available_ram_mb * (1 - safety_margin)

        if avg_paramset_size_mb <= 0: return min_batch

        z = int(usable_ram_mb // avg_paramset_size_mb)
        return max(min_batch, min(z, max_batch))

    # || ===================================================================== || Signals Functions || ===================================================================== ||

    def _calculate_indicator(self, model_timeframe, ind_name, ind_obj, param_set_dict, curr_asset_obj, asset_name, datetime_candle_references):        
        # Calcula indicadores com cache único por instância e alinhamento MTF seguro.
        
        # 1. RESOLUÇÃO DE PARÂMETROS
        effective_params = ind_obj.params.copy()
        if param_set_dict:
            for k, v in effective_params.items():
                if isinstance(v, str) and v in param_set_dict:
                    effective_params[k] = param_set_dict[v]

        target_asset = ind_obj.asset if ind_obj.asset else asset_name
        target_tf = ind_obj.timeframe if ind_obj.timeframe else model_timeframe
        
        # 2. CHAVE DE CACHE ÚNICA
        # Usamos o 'ind_name' (ex: 'ma_fast') + params para garantir que 
        # mesmo que 'ma_fast' e 'ma_slow' tenham janelas iguais em algum momento,
        # eles ocupem espaços distintos.
        params_str = "_".join([f"{k}_{v}" for k, v in sorted(effective_params.items())])
        cache_key = f"{ind_name}_{target_asset}_{target_tf}_{params_str}"

        # Inicializa cache do ativo se não existir
        if asset_name not in self._results_map: self._results_map[asset_name] = {}
        
        # Retorno rápido se já calculado nesta otimização
        if cache_key in self._results_map[asset_name]:
            print(f'> {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Using cached indicator: {ind_name} - Asset: {target_asset} - TF: {target_tf} - Params: {params_str}')
            return curr_asset_obj.with_columns([
                self._results_map[asset_name][cache_key].alias(ind_name)
            ])

        # 3. OBTENÇÃO DOS DADOS DE ORIGEM
        source_asset_class = self.assets.get(target_asset)
        if not source_asset_class:
            raise ValueError(f"Asset '{target_asset}' não carregado na operação.")

        df_source = source_asset_class.data_get(target_tf)

        # 4. CÁLCULO REAL
        print(f'> {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Calculating indicator: {ind_name} - Asset: {target_asset} - TF: {target_tf} - Params: {params_str}')
        indicator_result = ind_obj.calculate(df_source, param_set_dict=param_set_dict)

        # Garantir que temos uma Series
        if isinstance(indicator_result, pl.DataFrame):
            indicator_series = indicator_result.select(pl.all().exclude("datetime")).to_series()
        else:
            indicator_series = indicator_result

        # 5. VERIFICAÇÃO DE ALINHAMENTO (Sua dúvida do item 3)
        # Verificamos se precisamos de processamento MTF ou se é o mesmo timeframe
        is_same_asset = (target_asset == asset_name)
        is_same_tf = (target_tf == model_timeframe)

        if not (is_same_asset and is_same_tf):
            # Caso MTF ou Ativo Diferente: Alinhamento robusto com sua função
            print(f'> {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Aligning MTF indicator: {ind_name} - From Asset: {target_asset} - TF: {target_tf} To Asset: {asset_name} - TF: {model_timeframe}')
            df_temp_htf = pl.DataFrame({
                'datetime': df_source['datetime'].to_list() if hasattr(df_source['datetime'], 'to_list') else df_source['datetime'],
                ind_name: indicator_series.to_numpy() if hasattr(indicator_series, 'to_numpy') else indicator_series
            })

            aligned_df = self.transfer_htf_columns(
                ltf_df=curr_asset_obj.select(['datetime']), 
                ltf_tf=model_timeframe,
                htf_df=df_temp_htf,
                htf_tf=target_tf,
                datetime_reference_candles=datetime_candle_references,
                add_htf_tag=False 
            )
            final_series = aligned_df[ind_name].fill_null(strategy="forward")
        else:
            # Caso idêntico: Apenas alinhar índice e preencher gaps se houver
            # O reindex é necessário apenas se os shapes forem diferentes
            final_series = indicator_series
            # if len(indicator_series) != len(curr_asset_obj):
            #     final_series = indicator_series.reindex(curr_asset_obj.index, method='ffill')
            # else:
            #     final_series = indicator_series
            #     final_series.index = curr_asset_obj.index

        # 6. ATUALIZAÇÃO DO CACHE E DO DATAFRAME ATUAL
        self._results_map[asset_name][cache_key] = final_series
        
        return curr_asset_obj.with_columns([
            final_series.alias(ind_name)
        ])

    def _get_all_models(self) -> dict: # Returns all Model(s) from data
        if isinstance(self.data, Model): # Single Model
            return {self.data.name: self.data}
        elif isinstance(self.data, list): # List of Models
            all_models = {}
            for item in self.data:
                if isinstance(item, Model):
                    all_models[item.name] = item
                elif isinstance(item, dict):
                    all_models.update(item)
            return all_models
        else: return {}

    def _assets(self, target_asset_name: str, target_tf: str):
        # Recupera do contexto da classe
        actual_ltf_df = self._curr_df_context
        actual_model_tf = self._curr_tf_context
        actual_asset_name = self._curr_asset

        asset_obj = self.assets.get(target_asset_name)
        htf_df = asset_obj.data_get(target_tf) # Retorna pl.DataFrame
        
        # Se for o mesmo ativo e TF, retorna o DF de contexto
        if target_asset_name == actual_asset_name and target_tf == actual_model_tf:
            return actual_ltf_df
            
        # Usa a função original para alinhar
        return self.transfer_htf_columns(
            ltf_df=actual_ltf_df.select(["datetime"]), # Apenas datetime para o join
            ltf_tf=actual_model_tf,
            htf_df=htf_df,
            htf_tf=target_tf,
            datetime_reference_candles=self._curr_datetime_references,
            add_htf_tag=False
        )
    
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
        ltf_df: pl.DataFrame,
        ltf_tf: str,
        htf_df: pl.DataFrame,
        htf_tf: str,
        datetime_reference_candles: str = 'open',
        columns: Optional[List[str]] = None,
        add_htf_tag: bool = True
        ) -> pl.DataFrame:
        
        def get_tf_timedelta(tf: str):
            if tf.startswith('M') and not tf.startswith('MN'):
                return datetime.timedelta(minutes=int(tf[1:]))
            elif tf.startswith('H'):
                return datetime.timedelta(hours=int(tf[1:]))
            elif tf.startswith('D'):
                return datetime.timedelta(days=int(tf[1:]))
            return None

        # No Polars trabalhamos com seleções, não cópias manuais
        if columns is None:
            columns = [c for c in htf_df.columns if c != 'datetime']

        # Alinhamento de colunas HTF (renomeação)
        renamed_columns = {}
        for col in columns:
            if not add_htf_tag or col.endswith(f"_{htf_tf}"):
                renamed_columns[col] = col
            else:
                renamed_columns[col] = f"{col}_{htf_tf}"

        # Preparar HTF aligned: Seleciona e renomeia
        htf_aligned = htf_df.select(['datetime'] + columns).rename(renamed_columns)

        # Shift para evitar Look-ahead bias: O dado só existe após o fechamento da barra
        if datetime_reference_candles == 'open':
            delta = get_tf_timedelta(htf_tf)
            if delta:
                htf_aligned = htf_aligned.with_columns(
                    pl.col("datetime") + delta
                )
        elif datetime_reference_candles != 'close':
            raise ValueError("datetime_reference_candles must be 'open' or 'close'")

        # Join ASOF (ltf_df e htf_df devem estar ordenados por datetime)
        # Removemos colunas conflitantes do LTF antes (exceto datetime)
        cols_to_keep = [c for c in ltf_df.columns if c not in htf_aligned.columns or c == 'datetime']
        
        merged = ltf_df.select(cols_to_keep).join_asof(
            htf_aligned,
            on="datetime",
            strategy="backward" # Busca o último valor HTF disponível no tempo do LTF
        )

        return merged

    # || ===================================================================== || Save and Clean Functions || ===================================================================== ||

    def _print_metrics(self, key: str, trades: list):
        pass

    def _save_and_clean(self):
        pass

    # || ===================================================================== || Metrics Functions || ===================================================================== ||

    def _report_pnl_summary(self):
        print("\n" + "="*95)
        print(f"{'Performance Summary - Operation: ' + self.name:^95}")
        print("="*95)

        # Acessa o dicionário de modelos
        models = self._results_map.get(self.name, {}).get("models", {})
        
        if not models:
            print("No models found in results map.")
            return

        for model_name, model_data in models.items():
            print(f"\nModel: {model_name}")
            
            for strat_name, strat_data in model_data.get("strats", {}).items():
                print(f"  └── Strat: {strat_name}")
                
                for asset_name, asset_data in strat_data.get("assets", {}).items():
                    print(f"      └── Asset: {asset_name}")
                    
                    for param_key, param_data in asset_data.get("param_sets", {}).items():
                        trades = param_data.get("trades", [])
                        
                        if not trades:
                            print(f"          └── {param_key}: No trades.")
                            continue

                        # Função auxiliar para extrair valores (suporta dict ou objeto)
                        def get_val(t, attr):
                            return t.get(attr, 0) if isinstance(t, dict) else getattr(t, attr, 0)

                        # SEPARAÇÃO POR LADO (Baseado no sinal do lot_size)
                        longs = [t for t in trades if get_val(t, 'lot_size') > 0]
                        shorts = [t for t in trades if get_val(t, 'lot_size') < 0]

                        def calc_metrics(trade_list):
                            if not trade_list: 
                                return {"pnl": 0.0, "wr": 0.0, "cnt": 0, "avg": 0.0}
                            
                            p_list = [get_val(t, 'profit') for t in trade_list]
                            cnt = len(p_list)
                            pnl_sum = sum(p_list)
                            wins = len([p for p in p_list if p > 0])
                            
                            return {
                                "pnl": pnl_sum,
                                "wr": (wins / cnt) * 100 if cnt > 0 else 0,
                                "cnt": cnt,
                                "avg": pnl_sum / cnt if cnt > 0 else 0
                            }

                        m_all = calc_metrics(trades)
                        m_long = calc_metrics(longs)
                        m_short = calc_metrics(shorts)

                        print(f"          └── Param Set: {param_key}")
                        print(f"              {'-'*80}")
                        print(f"              {'METRICA':<15} | {'GERAL':<15} | {'COMPRA (L)':<15} | {'VENDA (S)':<15}")
                        print(f"              {'-'*80}")
                        print(f"              {'Total Trades':<15} | {m_all['cnt']:<15} | {m_long['cnt']:<15} | {m_short['cnt']:<15}")
                        print(f"              {'PnL %':<15} | {m_all['pnl']:>14.2f}% | {m_long['pnl']:>14.2f}% | {m_short['pnl']:>14.2f}%")
                        print(f"              {'Winrate':<15} | {m_all['wr']:>14.2f}% | {m_long['wr']:>14.2f}% | {m_short['wr']:>14.2f}%")
                        print(f"              {'Avg Trade':<15} | {m_all['avg']:>14.4f}% | {m_long['avg']:>14.4f}% | {m_short['avg']:>14.4f}%")
                        
                        all_pnls = [get_val(t, 'profit') for t in trades]
                        print(f"              {'-'*80}")
                        print(f"              Best Trade: {max(all_pnls):.2f}%  |  Worst Trade: {min(all_pnls):.2f}%\n")

        print("\n" + "="*95)

    def _plot_pnl_curves(self, mode: str = 'param_sets'):
        import matplotlib.pyplot as plt
        import polars as pl

        all_series = []
        # Acessa os modelos dentro do results_map
        models = self._results_map.get(self.name, {}).get("models", {})
        
        for m_name, m_data in models.items():
            for s_name, s_data in m_data.get("strats", {}).items():
                for a_name, a_data in s_data.get("assets", {}).items():
                    for p_name, p_data in a_data.get("param_sets", {}).items():
                        trades = p_data.get("trades", [])
                        if not trades: continue
                        
                        # Criar DataFrame local
                        # Garantimos que profit seja Float64 e datetime seja Datetime
                        df_trades = pl.DataFrame(trades).select([
                            pl.col("exit_datetime").str.to_datetime().alias("datetime"),
                            pl.col("profit").cast(pl.Float64)
                        ])
                        
                        # Agrupa lucro por datetime
                        df_trades = df_trades.group_by("datetime").agg(pl.col("profit").sum()).sort("datetime")
                        
                        serie_name = f"{s_name}_{a_name}_{p_name}" if mode == 'param_sets' else s_name
                        all_series.append(df_trades.rename({"profit": serie_name}))

        if not all_series:
            print("< Erro: Nenhum trade encontrado para plotagem.")
            return

        # 1. Alinhamento usando how='full' (substituindo o depreciado 'outer')
        consolidated = all_series[0]
        for i in range(1, len(all_series)):
            consolidated = consolidated.join(all_series[i], on="datetime", how="full", coalesce=True)

        # 2. Ordenação e Tratamento de nulos
        consolidated = consolidated.sort("datetime")

        # 3. Identifica apenas as colunas de PnL (exclui a coluna datetime)
        pnl_cols = [c for c in consolidated.columns if c != "datetime"]
        
        # CORREÇÃO DO ERRO: Cast explícito antes do fill_null e aplicação apenas nas colunas PnL
        consolidated = consolidated.with_columns([
            pl.col(c).cast(pl.Float64).fill_null(0.0).cum_sum().alias(c) 
            for c in pnl_cols
        ])

        # 4. Plotagem
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 6))

        # Converte para pandas para o matplotlib
        pdf = consolidated.to_pandas().set_index("datetime")
        
        # Preenchimento frontal (Forward Fill) para garantir as retas entre trades
        pdf = pdf.ffill().fillna(0.0)

        for col in pdf.columns:
            ax.plot(pdf.index, pdf[col], label=col, linewidth=1.5, alpha=0.8)

        ax.set_title(f"Cumulative PnL Curves - Operation: {self.name}", fontsize=14, color='gold', pad=20)
        ax.set_xlabel("Timeline", fontsize=10)
        ax.set_ylabel("Cumulative Profit (%)", fontsize=10)
        ax.legend(loc='upper left', fontsize='x-small', framealpha=0.2)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    # || ======================================================================================================================================================================= ||
                        
    def run(self):
        # I - Init and Validation of Operation
        print(f"\n>>> I - Init and Validating Operation <<<")
        self._validate_operation()

        # II - Data Pre-Processing and Execution
        print(f"\n>>> II - Data Pre-Processing, Calculating Param Sets, Indicators, Signals and Backtests <<<")
        self._operation()
        self._report_pnl_summary()
        self._plot_pnl_curves()

        # III - Operation Portfolio Simulation, Operation Analysis and Metrics
        print(f"\n>>> III - Operation Portfolio Simulation, Operation Analysis and Metrics <<<")
        self._simulate_portfolio()

        # IV - Pos-Processing, Saving and Cleaning
        print(f"\n>>> IV - Pos-Processing, Saving and Cleaning <<<")
        self._save_and_clean()

        return self._results_map



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


    model_assets=['EURUSD'] # Only keys #, 'GBPUSD'
    model_execution_tf = 'M15'

    Params = {
        'AT15': { 
            'execution_tf': model_execution_tf,
            'sl_perc': range(3, 3+1, 1), # 3
            'tp_perc': range(9, 9+1, 1), 
            'param1': range(20, 20+1, 30), #50
            'param2': range(2, 2+1, 1), # 3
            'param3': ['sma'] #, 'ema', 'ema'
        }
    }

    # User imput Indicators
    ind = { 
        'sma': MA(asset=None, timeframe=model_execution_tf, window='param1', ma_type='param3', price_col='close'),
        'ema_htf': MA(asset='GBPUSD', timeframe='D1', window=252, ma_type='ema') #, price_col='open'
    }

    def entry_long(self, df: pl.DataFrame, curr_param_set: dict): 
        df_D1 = self._assets(self._curr_asset, 'D1')
        df_EURUSD_D1 = self._assets('EURUSD', 'D1')

        sl_perc = curr_param_set['sl_perc']
        diff = df['close']*(sl_perc/100)
        ema_htf = df['ema_htf']

        signal = (df['close'] < df['open']) & (df['close'].shift(1) < df['open'].shift(1)) & (df['close'].shift(2) < df['open'].shift(2)) 
        #(df['close'] < df['sma']) & (df['sma'] != 0.0) # & (df['close'] > df['open']) #& df['close'].shift(1) < df['sma'].shift(1)) # (df['close'] > df['sma'] + diff) & (df['close'].shift(1) < df['sma'].shift(1) + diff) & (df_D1['close'] > df_D1['close'].shift(1))
        return signal
    
    def entry_short(self, df: pl.DataFrame, curr_param_set: dict): 
        df_D1 = self._assets(self._curr_asset, 'D1')
        df_EURUSD_D1 = self._assets('EURUSD', 'D1')

        sl_perc = curr_param_set['sl_perc']
        diff = df['close']*(sl_perc/100)
        ema_htf = df['ema_htf']

        signal = (df['close'] > df['open']) & (df['close'].shift(1) > df['open'].shift(1)) & (df['close'].shift(2) > df['open'].shift(2)) 
        #(df['close'] > df['sma']) & (df['sma'] != 0.0) # & (df['close'] < df['open']) #& df['close'].shift(1) > df['sma'].shift(1)) # (df['close'] < df['sma'] - diff) & (df['close'].shift(1) > df['sma'].shift(1) - diff) & (df_D1['close'] < df_D1['close'].shift(1))
        return signal

    def exit_tf_long(self, df: pl.DataFrame, curr_param_set: dict):
        return (df['close'] > df['open']) & (df['close'].shift(1) > df['open'].shift(1))
    
    def exit_tf_short(self, df: pl.DataFrame, curr_param_set: dict):
        return (df['close'] < df['open']) & (df['close'].shift(1) < df['open'].shift(1))

    """
    def exit_sl_long(self, df: pl.DataFrame, curr_param_set: dict):
        return df['close'] - (df['close']*(curr_param_set['sl_perc']/100))
    def exit_sl_short(self, df: pl.DataFrame, curr_param_set: dict):
        return df['close'] + (df['close']*(curr_param_set['sl_perc']/100))

    def exit_tp_long(self, df: pl.DataFrame, curr_param_set: dict):
        return df['close'] + (df['close']*(curr_param_set['tp_perc']/100))
    def exit_tp_short(self, df: pl.DataFrame, curr_param_set: dict):
        return df['close'] - (df['close']*(curr_param_set['tp_perc']/100))
    """
    exit_sl_long = None
    exit_sl_short = None
    exit_tp_long = None
    exit_tp_short = None

    AT15 = Strat(
        StratParams(
            name="AT15",
            operation=Backtest(BacktestParams(name='backtest_test')),
            execution_settings=ExecutionSettings(hedge=False, strat_num_pos=[1,1], order_type='market', offset=0.0),
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
                'exit_sl_long': exit_sl_long,
                'exit_sl_short': exit_sl_short,
                'exit_tp_long': exit_tp_long,
                'exit_tp_short': exit_tp_short,

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
            data=[model_1],
            #operation=Backtest(BacktestParams(name='backtest_test')),
            operation_backtest_all_signals_are_positions=False,
            assets=global_assets,
            operation_timeframe=model_execution_tf, # Must always be the smaller timeframe among all strat execution_timeframe
            date_start=None, #'2020-01-01',
            date_end=None, #'2023-01-01',
            save=False,
            metrics={}
        )
    )

    operation.run()


