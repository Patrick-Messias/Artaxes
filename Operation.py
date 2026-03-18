import polars as pl, numpy as np, json, sys, uuid, math, datetime, psutil, re, itertools
sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend\Indicators')
sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend')

from webbrowser import get
from typing import Union, Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict, is_dataclass
from Model import ModelParams, Model
from Asset import Asset, AssetParams
from Strat import Strat, StratParams, ExecutionSettings
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
                                    'wfm_matrix_data': list[list[Trade]], # Raw daily returns matrix from all param_sets
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
    }
}

indic_cache = {
    '{asset_name}': { # Indicators
        '{ind_name}': {
            '{cache_key}': pl.Series # self._results_map[asset_name][ind_name][cache_key]
        }
    }
}

# =========================================================================================================================================|| Global Mapping

@dataclass
class OperationParams():
    name: str = field(default_factory=lambda: f'model_{uuid.uuid4()}')
    data: Union[Model, list[Model]]=None # Can make an operation with a single model or portfolio
    assets: Optional[Dict[str, Any]] = field(default_factory=dict) # Global Assets

    # Metrics
    metrics: Optional[Dict[str, Indicator]] = field(default_factory=dict)

    # Settings
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

        self.operation_timeframe = op_params.operation_timeframe
        self.date_start = op_params.date_start
        self.date_end = op_params.date_end
        self.save = op_params.save

        self._results_map = {}  
        self._indicators_cache = {}
        self.unique_datetime_df = pl.DataFrame

        self._curr_asset: Optional[str] = None
        self._curr_df_context: Optional[pl.DataFrame] = None
        self._curr_tf_context: Optional[str] = None
        self._curr_datetime_references: Optional[str] = None

    # || ===================================================================== || I - Operation Validation || ===================================================================== ||

    def _validate_operation(self):
        pass

    # || ===================================================================== || II - Data Processing || ===================================================================== ||

    def _operation(self):
        models = self._get_all_models()
        self._results_map[self.name] = {'models': {}}
 
        for model_name, model_obj in models.items():
            strats    = model_obj.strat
            assets    = model_obj.assets
            model_tf  = model_obj.execution_timeframe
            self._results_map[self.name]['models'][model_name] = {'strats': {}}
 
            for strat_name, strat_obj in strats.items():
                param_sets = self._calculate_param_combinations(strat_obj.params)
                self._results_map[self.name]['models'][model_name]['strats'][strat_name] = {'assets': {}}
 
                for asset_name in assets:
                    asset_class = self.assets.get(asset_name)
                    if not asset_class: continue
 
                    self._results_map[self.name]['models'][model_name]['strats'][strat_name]['assets'][asset_name] = {'param_sets': {}}
                    base_asset_df  = asset_class.data_get(model_tf)
                    exec_set_raw   = asdict(strat_obj.execution_settings)
                    exec_set_mod   = self.prepare_time_params(exec_set_raw)
                    n_ps           = len(param_sets)
 
                    ind_cache:   dict[str, dict] = {}
                    ps_ind_keys: dict[str, list] = {}
                    sig_cache:   dict[str, dict] = {}
                    ps_sig_hash: dict[str, str] = {}
 
                    import time
                    _t = time.perf_counter()
                    # ── Fase 1: Indicadores e Sinais ───────────────────────────
                    sig_key_params: list | None = None

                    for ps_name, ps_dict in param_sets.items():
                        ps_ind_keys[ps_name] = []
 
                        for ind_key, ind_obj in strat_obj.indicators.items():
                            eff_params = self.effective_params_from_global(ind_obj.params, ps_dict)
                            ind_p_hash = self.param_suffix(eff_params)
                            unique_key = f"{asset_name}_{model_tf}_{ind_key}_{ind_p_hash}"
 
                            if unique_key not in ind_cache:
                                print(f"   > Calculating indicator >>> {ind_key} >>> {unique_key}")
                                temp_ind_df   = self._calculate_indicator(
                                    model_timeframe=model_tf,
                                    ind_name=ind_key,
                                    ind_obj=ind_obj,
                                    param_set_dict=ps_dict,
                                    curr_asset_obj=base_asset_df,
                                    asset_name=asset_name,
                                    datetime_candle_references=asset_class.datetime_candle_references
                                )
                                novas_colunas = [c for c in temp_ind_df.columns if c not in base_asset_df.columns]
                                ind_cache[unique_key] = {
                                    c: temp_ind_df[c].cast(pl.Float64).fill_null(0.0).to_numpy().astype(np.float64)
                                    for c in novas_colunas
                                }
                            #else:
                            #    print(f"   > Using cache indicator >>> {ind_key} >>> {unique_key}")
 
                            ps_ind_keys[ps_name].append(unique_key)

                        # Determines the sig_hash for this param_set, if __sig_key_params is declared, use reducted hash
                        if sig_key_params is not None:
                            sig_hash = self.param_suffix({k: ps_dict[k] for k in sig_key_params if k in ps_dict})
                        else:
                            sig_hash = self.param_suffix(ps_dict)
                        ps_sig_hash[ps_name] = sig_hash
                        
                        if sig_hash not in sig_cache:
                            df_full = base_asset_df.clone()
                            for uk in ps_ind_keys[ps_name]:
                                for col_name, values in ind_cache[uk].items():
                                    df_full = df_full.with_columns(pl.Series(col_name, values))
 
                            sig_result = strat_obj.signals(df_full, ps_dict)
                            sig_cache[sig_hash] = {}
 
                            # Detects __sig_key_params in first call
                            if sig_key_params is None and '__sig_key_params' in sig_result:
                                sig_key_params = sig_result['__sig_key_params']

                                # Recalculates sig_hash with reduced params
                                sig_hash = self.param_suffix({k: ps_dict[k] for k in sig_key_params if k in ps_dict})
                                ps_sig_hash[ps_name] = sig_hash

                                # Moves results to correct hash
                                sig_result_copy = {k: v for k, v in sig_result.items() if k != '__sig_key_params'}
                                sig_cache[sig_hash] = {}
                                sig_result = sig_result_copy

                            for sig_name, sig_val in sig_result.items():
                                if sig_name == '__sig_key_params': continue
                                if sig_val is None: continue
                                if isinstance(sig_val, str):
                                    sig_cache[sig_hash][sig_name] = sig_val
                                    continue
                                if isinstance(sig_val, pl.Expr):
                                    sig_val = df_full.select(sig_val).to_series()
                                if isinstance(sig_val, pl.Series):
                                    dtype = sig_val.dtype
                                    if dtype == pl.Boolean or dtype == pl.UInt8:
                                        sig_cache[sig_hash][sig_name] = (
                                            sig_val.cast(pl.Boolean).fill_null(False)
                                            .to_numpy().astype(np.uint8)
                                        )
                                    else :
                                        sig_cache[sig_hash][sig_name] = (
                                            sig_val.cast(pl.Float64).fill_null(0.0)
                                            .to_numpy().astype(np.float64)
                                        )
                                elif isinstance(sig_val, np.ndarray):
                                    if sig_val.dtype == np.uint8 or sig_val.dtype == bool:
                                        sig_cache[sig_hash][sig_name] = sig_val.astype(np.uint8)
                                    else:
                                        sig_cache[sig_hash][sig_name] = sig_val.astype(np.float64)
                                elif isinstance(sig_val, list):
                                    sample = next((v for v in sig_val if v is not None), 0)
                                    if isinstance(sample, bool):
                                        sig_cache[sig_hash][sig_name] = np.asarray(sig_val, dtype=np.uint8)
                                    else:
                                        sig_cache[sig_hash][sig_name] = np.asarray(sig_val, dtype==np.float64)
                        else: pass

                    print(f"   > [DEBUG] sig_cache unique hashes: {len(sig_cache)} / {n_ps} param_sets")
                    print(f"   > [OP] fase1={time.perf_counter()-_t:.2f}s"); _t = time.perf_counter()
                    # ── Fase 2: Indicators Pool ────────────────────────────────
                    indicators_pool:  dict = {}
                    ps_ind_col_keys:  dict[str, list] = {}
 
                    for ps_name, ps_dict in param_sets.items():
                        ps_ind_col_keys[ps_name] = []
                        for uk in ps_ind_keys[ps_name]:
                            for col_name, values in ind_cache[uk].items():
                                pool_key = f"{uk}__{col_name}"
                                if pool_key not in indicators_pool:
                                    indicators_pool[pool_key] = values
                                ps_ind_col_keys[ps_name].append(pool_key)
                    print(f"   > [OP] fase2={time.perf_counter()-_t:.2f}s"); _t = time.perf_counter()
                    # ── Fase 3: Arrays de preço derivados → pool ──────────────
                    # pl.Series float retornadas pelo strat_signals entram no pool
                    # com key "sig__{sig_hash}__{sig_name}" (única por param_set).
                    # Strings apontam para coluna já existente (ohlc ou pool) → signal_ref.
                    # Arrays uint8 → signal_array (entry/exit binários).
                    ohlc_col_names = set(base_asset_df.columns)

                    ps_signal_refs:   dict[str, dict] = {ps: {} for ps in param_sets}
                    ps_signal_arrays: dict[str, dict] = {ps: {} for ps in param_sets}

                    for ps_name, ps_dict in param_sets.items():
                        sig_hash = ps_sig_hash[ps_name]  # usa hash reduzido se disponível
                        for sig_name, sig_val in sig_cache[sig_hash].items():
                            if isinstance(sig_val, str):
                                ps_signal_refs[ps_name][sig_name] = sig_val
                            elif isinstance(sig_val, np.ndarray):
                                if sig_val.dtype == np.uint8:
                                    ps_signal_arrays[ps_name][sig_name] = sig_val
                                else:
                                    pool_key = f"sig__{sig_hash}__{sig_name}"
                                    if pool_key not in indicators_pool:
                                        indicators_pool[pool_key] = sig_val
                                    ps_signal_refs[ps_name][sig_name] = pool_key
                    print(f"   > [OP] fase3={time.perf_counter()-_t:.2f}s"); _t = time.perf_counter()
                    # ── Fase 4: Shared signal arrays ───────────────────────────
                    all_sig_names: set = set()
                    for d in ps_signal_arrays.values(): all_sig_names.update(d.keys())
 
                    shared_signal_arrays: dict = {}
                    for sig_name in all_sig_names:
                        arrs = [ps_signal_arrays[ps].get(sig_name) for ps in param_sets
                                if ps_signal_arrays[ps].get(sig_name) is not None]
                        if len(arrs) == len(param_sets):
                            first = arrs[0]
                            if all(a is first or np.array_equal(a, first) for a in arrs[1:]):
                                shared_signal_arrays[sig_name] = first
                                for ps in param_sets:
                                    ps_signal_arrays[ps].pop(sig_name, None)
 
                    print(f"   > Pool: {len(indicators_pool)} cols | "
                          f"Shared bin: {len(shared_signal_arrays)} | Param_sets: {n_ps}")
                    print(f"   > [OP] fase4={time.perf_counter()-_t:.2f}s"); _t = time.perf_counter()
                    # ── Fase 5: Monta asset_batch ──────────────────────────────
                    asset_batch = {
                        "asset_header":         f"{model_name}_{strat_name}_{asset_name}",
                        "data":                 base_asset_df.to_dict(as_series=False),
                        "execution_settings":   exec_set_mod,
                        "indicators_pool":      indicators_pool,
                        "shared_signal_arrays": shared_signal_arrays,
                        "simulations":          [],
                    }
 
                    for ps_name, ps_dict in param_sets.items():
                        self._results_map[self.name]['models'][model_name]['strats'][strat_name]['assets'][asset_name]['param_sets'][ps_name] = {
                            'param_set_dict': ps_dict,
                            'trades': []
                        }
                        asset_batch["simulations"].append({
                            "id":             ps_name,
                            "params":         ps_dict,
                            "indicator_keys": ps_ind_col_keys[ps_name],
                            "signal_arrays":  ps_signal_arrays[ps_name],
                            "signal_refs":    ps_signal_refs[ps_name],
                        })
 
                    # Estimates batch size — hybrid CPU+RAM approach
                    ps_size_mb = self._estimate_paramset_size_mb(base_asset_df) / max(n_ps, 1)
                    batch_size = self._calculate_optimal_batch_size(
                        avg_paramset_size_mb=ps_size_mb,
                        safety_margin=0.6,  # RAM: use 40% available
                        max_batch=n_ps,
                        min_batch=1
                    )
                    all_sim   = list(asset_batch.pop("simulations"))
                    n_batches = math.ceil(n_ps / batch_size)
                    print(f"   > Batching: {n_ps} sims | batch_size={batch_size} | n_batches={n_batches}")
 
                    # ── Fase 6: Envio para C++ ─────────────────────────────────
                    all_ps_names = [s.get("id", "") for s in all_sim]
                    wfm_accum = {"ts": [], "pnl": [], "lot_size": [], "ps_id": []}
                    for batch_start in range(0, n_ps, batch_size):
                        batch_sims = all_sim[batch_start:batch_start + batch_size]
                        asset_batch["simulations"] = batch_sims
                        full_output = self._run_cpp_operation(asset_batch)
                        self._save_trades(full_output, model_name, strat_name, asset_name,
                                          wfm_accum=wfm_accum,
                                          ps_id_offset=batch_start)
                    
                    # Single pivot after all batches
                    if any(len(v) > 0 for v in wfm_accum.values()):
                        try:
                            wfm_col = {
                                "ts": np.concatenate(wfm_accum["ts"]),
                                "pnl": np.concatenate(wfm_accum["pnl"]),
                                "lot_size": np.concatenate(wfm_accum["lot_size"]),
                                "ps_id": np.concatenate(wfm_accum['ps_id'])
                            }
                            df_matrix = self._process_wfm_to_polars(wfm_col, all_ps_names )
                            self._results_map[self.name]["models"][model_name]["strats"][strat_name]["assets"][asset_name]["wfm_matrix_data"] = df_matrix
                            print(f"   > WFM Matrix generated for {model_name} | {strat_name} | {asset_name}: {df_matrix.shape}")
                        except Exception as e:
                            print(f"   > Error generating WFM Matrix: {e}")
                    print(f"   > [OP] fase5_prep={time.perf_counter()-_t:.2f}s"); _t = time.perf_counter()
        return True
    
    def _calculate_indicator(self, model_timeframe, ind_name, ind_obj, param_set_dict, curr_asset_obj, asset_name, datetime_candle_references):        
        final_output = None

        # 1. Resolução de Alvos (Asset e Timeframe onde o indicador "mora")
        target_asset = ind_obj.asset if ind_obj.asset else asset_name
        target_tf = ind_obj.timeframe if ind_obj.timeframe else model_timeframe

        # Validação de Timeframe (Garante que não tentamos olhar o futuro/LTF)
        if self._tf_to_seconds(target_tf) < self._tf_to_seconds(model_timeframe):
            raise ValueError(f"Indicador '{ind_name}' ({target_tf}) não pode ser menor que o TF da Estratégia ({model_timeframe}).")

        # 2. Setup do Cache de Dados BRUTOS (Raw)
        effective_params = self.effective_params_from_global(ind_obj.params, param_set_dict)
        params_suffix = self.param_suffix(effective_params)

        if not hasattr(self, '_indicators_cache'): 
            self._indicators_cache = {}
        
        # Estrutura: self._indicators_cache[Ativo][Nome_Ind][TF_Ind][Param_Hash]
        tf_cache = self._indicators_cache.setdefault(target_asset, {}).setdefault(ind_name, {}).setdefault(target_tf, {})

        # --- CAMADA 1: Recuperar ou Calcular o dado no TF original ---
        if params_suffix in tf_cache:
            raw_result = tf_cache[params_suffix]
        else:
            source_asset_class = self.assets.get(target_asset)
            if not source_asset_class:
                raise ValueError(f"Ativo {target_asset} não encontrado nos ativos globais.")
            
            df_source = source_asset_class.data_get(target_tf)
            
            # Cálculo Real (Chama a lógica matemática do indicador)
            calc_res = ind_obj.calculate(df_source, param_set_dict=param_set_dict, ind_name=ind_name)

            # Normalização do Raw: Garante que temos um DataFrame com a coluna 'datetime' para o join
            if isinstance(calc_res, pl.Series):
                raw_result = pl.DataFrame({
                    "datetime": df_source["datetime"], 
                    ind_name: calc_res.fill_null(0.0)
                })
            else:
                raw_result = calc_res.fill_null(0.0)
                if "datetime" not in raw_result.columns:
                    raw_result = raw_result.with_columns(df_source["datetime"])

            # SALVA NO CACHE: Guardamos o dado no TF nativo (Ex: Diário)
            tf_cache[params_suffix] = raw_result

        # --- CAMADA 2: Alinhamento (Sincronização HTF -> LTF) ---
        is_same_asset = (target_asset == asset_name)
        is_same_tf = (target_tf == model_timeframe)

        if not (is_same_asset and is_same_tf):
            # Se o indicador for de outro ativo ou timeframe maior, precisamos alinhar
            # Ex: Pegar SMA(200) do Diário e expandir para cada minuto do intraday
            print(f'      > Synchronizing: {ind_name} ({target_asset} {target_tf}) -> {asset_name} ({model_timeframe})')
            
            # Criamos um esqueleto apenas com o datetime do gráfico atual (LTF)
            temp_align_df = curr_asset_obj.select("datetime")
            
            # Transferimos todas as colunas do indicador (exceto datetime) para o LTF
            for col in raw_result.columns:
                if col == "datetime": continue
                
                # Preparamos o dado HTF para o transfer_htf_columns
                htf_data_to_align = raw_result.select(["datetime", col])
                temp_align_df = self.transfer_htf_columns(temp_align_df, htf_data_to_align, col)
            
            # O resultado final não deve conter o datetime, apenas os valores alinhados
            final_output = temp_align_df.drop("datetime")
        else:
            # Caso seja o mesmo ativo e TF, apenas removemos colunas de sistema/ohlc
            # para evitar duplicidade no payload do C++
            cols_to_drop = [c for c in raw_result.columns if c in ['datetime', 'date', 'ativo', 'open', 'high', 'low', 'close', 'tick_volume', 'real_volume', 'spread']]
            final_output = raw_result.drop(cols_to_drop)

        # 3. Limpeza Final e Retorno
        # fill_nan(0) é importante pois joins HTF podem gerar NaNs no início do histórico
        return final_output.fill_nan(0.0).fill_null(0.0)

    # || ===================================================================== || III - Portfolio Simulator || ===================================================================== ||

    def _simulate_portfolio(self, portfolio_backtests_dict ='All'):
        # 1. Selects over all models -> strats -> assets -> param_sets, while iterating verifies if any walkforward operation is present
        # 2. Either selects 1 specific param_set for each strat/asset, iterates over all param_sets or select wf_param_set and iterates over walkforward param_sets
        # 3. Selects all unique datetime from selected backtest trades
        # 4. Iterates over datetimes, ranks param_sets based on previous trades results with some metric (ex: equity, profit factor, etc)
        # 5. For each datetime checks for entries and exits on each strat/asset/param_set simulating a portfolio with real money management and trade management rules
        #envelope theorem?


        if portfolio_backtests_dict == 'All': # Uses all backtests from _results_map, else if dict with paths, uses only those backtests 
            pass

        pass

    def _operation_analysis(self):
        pass

    # || ===================================================================== || Execution Functions || ===================================================================== ||

    def _run_cpp_operation(self, asset_batch: dict):
        try:
            path_to_dll = r"C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend\build\Release"
            if path_to_dll not in sys.path:
                sys.path.append(path_to_dll)
            import engine_cpp  # type: ignore

            df = pl.DataFrame(asset_batch['data']) if isinstance(asset_batch['data'], dict) \
                 else asset_batch['data'].clone()

            if df.schema.get('datetime') != pl.Int64:
                dt_int = (
                    df['datetime'].cast(pl.Datetime)
                    .dt.strftime('%Y%m%d%H%M%S')
                    .cast(pl.Int64)
                    .to_numpy()
                    .astype(np.int64)
                )
            else:
                dt_int = df['datetime'].to_numpy().astype(np.int64)

            ohlc_arrays = {
                col: df[col].to_numpy().astype(np.float64)
                for col in ['open', 'high', 'low', 'close'] if col in df.columns
            }

            ind_pool_arrays = {}
            for key, vals in asset_batch.get('indicators_pool', {}).items():
                if isinstance(vals, np.ndarray) and vals.dtype == np.float64:
                    ind_pool_arrays[key] = vals
                elif isinstance(vals, np.ndarray):
                    ind_pool_arrays[key] = vals.astype(np.float64)
                else:
                    ind_pool_arrays[key] = np.asarray(vals, dtype=np.float64)

            shared_sig_arrays = {
                name: arr.astype(np.uint8) if arr.dtype != np.uint8 else arr
                for name, arr in asset_batch.get('shared_signal_arrays', {}).items()
            }

            sim_params = []
            for sim in asset_batch.get('simulations', []):
                sig_arrays = {}
                for name, arr in sim.get('signal_arrays', {}).items():
                    sig_arrays[name] = arr.astype(np.uint8) if arr.dtype != np.uint8 else arr
                sim_params.append({
                    "id":            sim.get("id", ""),
                    "params":        sim.get("params", {}),
                    "signal_arrays": sig_arrays,
                    "signal_refs":   sim.get("signal_refs", {}),
                })

            ps_names = [s.get("id", "") for s in asset_batch.get("simulations", [])]
            raw_output = engine_cpp.execute(
                asset_batch.get('asset_header', 'Unknown'),
                ohlc_arrays,
                dt_int,
                ind_pool_arrays,
                shared_sig_arrays,
                sim_params,
                asset_batch.get('execution_settings', {}),
            )
            if not raw_output:
                return {"trades_columnar": None, "wfm_columnar": None, "ps_names": []}

            return {
                "trades_columnar": raw_output.get("trades_columnar"),
                "wfm_columnar":    raw_output.get("wfm_columnar"),
                "ps_names":        ps_names,
            }

        except Exception as e:
            print(f'< Error in Python-C++ Bridge: {e}')
            import traceback; traceback.print_exc()
            return {"trades_columnar": None, "wfm_columnar": None, "ps_names": []}


    def _save_trades(self, full_output: dict, m_name: str, s_name: str, a_name: str, wfm_accum: dict=None, ps_id_offset = None):
        if not full_output: return

        import time
        t0 = time.perf_counter()
 
        tc       = full_output.get("trades_columnar")
        wfm_col  = full_output.get("wfm_columnar")
        ps_names = full_output.get("ps_names", []) 
 
        # Trades: columnar -> list[dict] by simulation via polars
        if tc:
            # Builds DataFrame flat of all trades at unce
            df_all = pl.DataFrame({
                "entry_price":   tc["entry_price"],
                "exit_price":    tc["exit_price"],
                "lot_size":      tc["lot_size"],
                "stop_loss":     tc["stop_loss"],
                "take_profit":   tc["take_profit"],
                "profit":        tc["profit"],
                "profit_r":      tc["profit_r"],
                "mfe":           tc["mfe"],
                "mae":           tc["mae"],
                "bars_held":     tc["bars_held"],
                "closed":        tc["closed"],
                "id":            tc["id"],
                "entry_datetime":tc["entry_datetime"],
                "exit_datetime": tc["exit_datetime"],
                "exit_reason":   tc["exit_reason"],
                "status":        tc["status"],
            }).with_columns(pl.lit(a_name).alias("asset"))

            offsets = tc["sim_offsets"] # np.int32 [n_sims+1]
            t1 = time.perf_counter()

            for sim_idx, ps_name in enumerate(ps_names):
                if not ps_name: continue
                try:
                    target = self._results_map[self.name]["models"][m_name]["strats"][s_name]["assets"][a_name]["param_sets"][ps_name]
                    s = int(offsets[sim_idx])
                    e = int(offsets[sim_idx + 1])

                    # Saves pl.DataFrame directly without to_dicts()
                    target["trades"] = df_all.slice(s, e-s) if e>s else pl.DataFrame() 
                except KeyError as ex:
                    print(f"DEBUG: key not found: {ex} | {m_name}->{s_name}->{a_name}->{ps_name}")

            t2 = time.perf_counter()

        # Process WFM Daily Results, columnar -> _process_wfm_to_polars
        if wfm_col is not None and wfm_accum is not None:
            ts    = wfm_col.get("ts")
            pnl   = wfm_col.get("pnl")
            lot_size   = wfm_col.get("lot_size")
            ps_id = wfm_col.get("ps_id")

            if ts is not None and len(ts) > 0:
                wfm_accum["ts"].append(ts)
                wfm_accum["pnl"].append(pnl)
                wfm_accum["lot_size"].append(lot_size)
                wfm_accum["ps_id"].append(ps_id + ps_id_offset) # Ajusta ps_id pelo offset do batch


    def _process_wfm_to_polars(self, wfm_col: dict, ps_names: list = None):
        df = pl.DataFrame({
            "ts":    pl.Series(wfm_col["ts"],    dtype=pl.Int64),
            "pnl":   pl.Series(wfm_col["pnl"],   dtype=pl.Float64),
            "lot_size":   pl.Series(wfm_col["lot_size"],   dtype=pl.Float64),
            "ps_id": pl.Series(wfm_col["ps_id"], dtype=pl.Int32),
        })

        df = df.with_columns(
            pl.col("ts").cast(pl.Utf8).str.to_datetime("%Y%m%d%H%M%S")
        )

        matrix = df.pivot(
            on="ps_id",
            index="ts",
            values="pnl",
            aggregate_function="sum"
        ).sort("ts").fill_null(0.0)
        
        # Rename columns to ps_0, ps_1, ..., ps_{col}.
        # matrix = matrix.rename({
        #     col: f"ps_{col}" for col in matrix.columns if col != "ts"
        # })

        # Renames whole ps_id -> ps_name real if available
        if ps_names:
            # ps_id inteiro → "ps_{ps_name}" ex: 0 → "ps_param_set-21-8-sma-2"
            rename_map = {
                str(i): f"ps_{ps_names[i]}"
                for i in range(len(ps_names))
                if str(i) in matrix.columns
            }
            if rename_map:
                matrix = matrix.rename(rename_map)
        else:
            matrix = matrix.rename({
                col: f"ps_{col}" for col in matrix.columns if col != "ts"
            })

        return matrix

    # Batch System Defs
    def _estimate_paramset_size_mb(self, df: pl.DataFrame):
        return df.estimated_size() / (1024 ** 2) # No Polars, estimated_size() retorna o tamanho em bytes
    
    def _get_available_memory_mb(self):
        return psutil.virtual_memory().available / (1024 ** 2)
    
    def _calculate_optimal_batch_size(self, avg_paramset_size_mb, safety_margin=0.6, max_batch=1000, min_batch=1):
        import os
        available_ram_mb = self._get_available_memory_mb()
        usable_ram_mb    = available_ram_mb * (1.0 - safety_margin)
        n_cores          = os.cpu_count() or 4
 
        # RAM-based limit
        ram_limit = int(usable_ram_mb // avg_paramset_size_mb) if avg_paramset_size_mb > 0 else max_batch
 
        # CPU cache-based: 2x cores keeps L3 saturated without thrashing
        cache_limit = n_cores * 2
 
        z = min(ram_limit, cache_limit)
        return max(min_batch, min(z, max_batch))
 
    def prepare_time_params(self, settings: dict) -> dict:
        """
        Converte strings 'HH:MM' para minutos totais desde a meia-noite.
        Garante valores sentinela para o C++ caso os campos sejam None.
        """
        def to_min(t):
            if t is None or t == "" or t is False:
                return None
            try:
                # Lida com formatos HH:MM ou HH:MM:SS
                parts = str(t).split(':')
                h, m = int(parts[0]), int(parts[1])
                return h * 60 + m
            except Exception:
                return None
 
        # EI (Entry Initial): Default 0 (00:00)
        # EF (Entry Final): Default 1440 (24:00) - Não bloqueia novas entradas
        # TF (Time Finish): Default 1440 (24:00) - Não força fechamento
        
        return {
            **settings, # Mantém as outras configurações (is_daytrade, etc)
            "timeEI": to_min(settings.get("timeEI")) if settings.get("timeEI") is not None else 0,
            "timeEF": to_min(settings.get("timeEF")) if settings.get("timeEF") is not None else 1440,
            "timeTF": to_min(settings.get("timeTF")) if settings.get("timeTF") is not None else 1440
    }

    # || ===================================================================== || Signals Functions || ===================================================================== ||

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

   
    def transfer_htf_columns(self, ltf_df, htf_df, ind_name):
        
        # Alinha dados de ativos diferentes ou timeframes diferentes.
        # ltf_df: O DataFrame de destino (EURUSD M15).
        # htf_df: O DataFrame de origem (US30 H1).
        
        # 1. Limpeza e Ordenação (Essencial para join_asof)
        ltf_df = ltf_df.sort("datetime")
        htf_df = htf_df.select(["datetime", ind_name]).sort("datetime")

        # 2. Join Asof (Sincronização de Relógio)
        # 'backward' garante que o LTF só veja o valor do HTF que já aconteceu
        aligned = ltf_df.join_asof(
            htf_df,
            on="datetime",
            strategy="backward"
        )

        # 3. Cálculo de Integridade (Dados cruzados)
        total = len(aligned)
        nulos = aligned[ind_name].null_count()
        success_rate = ((total - nulos) / total) * 100
        
        # 4. Tratamento de Gaps e Início de Histórico
        # forward: preenche feriados/gaps no meio
        # backward: se o indicador começou depois do ativo da strat, preenche o início com o 1º valor
        aligned = aligned.with_columns(
            pl.col(ind_name)
            .fill_null(strategy="forward")
            .fill_null(strategy="backward")
            .fill_null(0.0)
        )

        print(f"      > Aligning {ind_name}: {success_rate:.2f}% covered.")
        if success_rate < 85:
            print(f"   ⚠️ WARNING: Only {success_rate:.2f}% of success aligning data.")

        return aligned
    
    def _tf_to_seconds(self, tf: str) -> int:
        
        #Converte strings de timeframe (ex: 'M15', '15m', 'H1', '1h') para segundos.
        
        tf_clean = tf.lower().strip()
        
        # Dicionário de multiplicadores
        multipliers = {
            's': 1,
            'm': 60,
            'h': 3600,
            'd': 86400,
            'w': 604800,
            'mo': 2592000
        }

        # Extrai apenas a parte numérica e a parte textual (unidade)
        # Ex: 'M15' -> unit='m', value=15 | '15m' -> unit='m', value=15
        match = re.search(r'([a-z]+)', tf_clean)
        unit = match.group(1) if match else None
        
        num_match = re.search(r'(\d+)', tf_clean)
        value = int(num_match.group(1)) if num_match else 1 # Default 1 se for apenas 'D', 'H', etc.

        if unit not in multipliers:
            # Tenta mapear abreviações comuns se necessário
            if unit == 'mn': unit = 'mo' # Mês no MT5 às vezes é MN
            else:
                raise ValueError(f"Unidade de tempo não suportada: {unit} no timeframe {tf}")

        return value * multipliers[unit]

    # || ===================================================================== || Save and Clean Functions || ===================================================================== ||

    def _print_metrics(self, key: str, trades: list):
        pass

    # Saves Model-Strat-Asset-Parset/WF results
    def _save_and_clean(self):
        pass

    # || ===================================================================== || Metrics Functions || ===================================================================== ||

    def _report_pnl_summary(self):
        print("\n" + "="*95)
        print(f"{'Performance Summary - Operation: ' + self.name:^95}")
        print("="*95)

        models = self._results_map.get(self.name, {}).get("models", {})
        if not models:
            print("No models found in results map.")
            return

        for model_name, model_data in models.items():
            print(f"\nModel: {model_name}")
            for strat_name, strat_data in model_data.get("strats", {}).items():
                print(f"  └── Strat: {strat_name}")
                for asset_path, asset_data in strat_data.get("assets", {}).items():
                    display_name = asset_path.split('_')[-1]
                    print(f"      └── Asset: {display_name}")

                    for param_key, param_data in asset_data.get("param_sets", {}).items():
                        trades = param_data.get("trades")

                        if trades is None or (isinstance(trades, pl.DataFrame) and trades.is_empty()) or \
                           (isinstance(trades, list) and len(trades) == 0):
                            print(f"          └── {param_key}: No trades.")
                            continue

                        # Normaliza para pl.DataFrame
                        if isinstance(trades, list):
                            df = pl.DataFrame(trades)
                        else:
                            df = trades

                        def calc_metrics(mask=None):
                            sub = df.filter(mask) if mask is not None else df
                            if sub.is_empty():
                                return {"pnl": 0.0, "wr": 0.0, "cnt": 0, "avg": 0.0}
                            cnt     = len(sub)
                            pnl_sum = sub["profit"].sum()
                            wins    = (sub["profit"] > 0).sum()
                            return {
                                "pnl": pnl_sum,
                                "wr":  (wins / cnt) * 100,
                                "cnt": cnt,
                                "avg": pnl_sum / cnt,
                            }

                        m_all   = calc_metrics()
                        m_long  = calc_metrics(pl.col("lot_size") > 0)
                        m_short = calc_metrics(pl.col("lot_size") < 0)
                        best    = df["profit"].max()
                        worst   = df["profit"].min()

                        print(f"          └── Param Set: {param_key}")
                        print(f"              {'-'*80}")
                        print(f"              {'METRICA':<15} | {'GERAL':<15} | {'COMPRA (L)':<15} | {'VENDA (S)':<15}")
                        print(f"              {'-'*80}")
                        print(f"              {'Total Trades':<15} | {m_all['cnt']:<15} | {m_long['cnt']:<15} | {m_short['cnt']:<15}")
                        print(f"              {'PnL %':<15} | {m_all['pnl']:>14.2f}% | {m_long['pnl']:>14.2f}% | {m_short['pnl']:>14.2f}%")
                        print(f"              {'Winrate':<15} | {m_all['wr']:>14.2f}% | {m_long['wr']:>14.2f}% | {m_short['wr']:>14.2f}%")
                        print(f"              {'Avg Trade':<15} | {m_all['avg']:>14.4f}% | {m_long['avg']:>14.4f}% | {m_short['avg']:>14.4f}%")
                        print(f"              {'-'*80}")
                        print(f"              Best Trade: {best:.2f}%  |  Worst Trade: {worst:.2f}%\n")

        print("\n" + "="*95)

    def _plot_pnl_curves(self, mode: str = 'param_sets'):
        import matplotlib.pyplot as plt

        all_series = []
        models = self._results_map.get(self.name, {}).get("models", {})

        for m_name, m_data in models.items():
            for s_name, s_data in m_data.get("strats", {}).items():
                for a_name, a_data in s_data.get("assets", {}).items():
                    for p_name, p_data in a_data.get("param_sets", {}).items():
                        trades = p_data.get("trades")
                        if trades is None: continue

                        if isinstance(trades, list):
                            if not trades: continue
                            df = pl.DataFrame(trades)
                        else:
                            if trades.is_empty(): continue
                            df = trades

                        # Filtra só trades fechados com exit_datetime
                        df = df.filter(pl.col("exit_datetime").is_not_null())
                        if df.is_empty(): continue

                        df_trades = df.select([
                            pl.col("exit_datetime").str.to_datetime("%Y%m%d %H%M%S").alias("datetime"),
                            pl.col("profit").cast(pl.Float64)
                        ])
                        df_trades = df_trades.group_by("datetime").agg(
                            pl.col("profit").sum()
                        ).sort("datetime")

                        serie_name = f"{s_name}_{a_name}_{p_name}" if mode == 'param_sets' else s_name
                        all_series.append(df_trades.rename({"profit": serie_name}))

        if not all_series:
            print("< Erro: Nenhum trade encontrado para plotagem.")
            return

        # Alinhamento
        from functools import reduce
        combined = reduce(lambda a, b: a.join(b, on="datetime", how="full", coalesce=True), all_series)
        combined = combined.sort("datetime").fill_null(0.0)
        cum_cols = [c for c in combined.columns if c != "datetime"]
        combined = combined.with_columns([
            pl.col(c).cum_sum().alias(c) for c in cum_cols
        ])

        pdf = combined.to_pandas()
        pdf.set_index("datetime", inplace=True)
        plt.figure(figsize=(14, 6))
        nums = 0
        for col in pdf.columns:
            nums +=1 
            plt.plot(pdf.index, pdf[col], label=col, linewidth=0.8)
        plt.title(f"PnL Curves — {self.name}")
        plt.xlabel("Date")
        plt.ylabel("Cumulative PnL %")
        print(nums)
        if nums < 41: plt.legend(fontsize=2, ncol=4)
        plt.tight_layout()
        plt.show()

    # || ===================================================================== || Walkforward || ===================================================================== ||

    def _run_walkforward(self): # Executes WFM for all Models -> Strats -> Assets
        import builtins
        models_dict = self._get_all_models()

        for m_name, m_obj in models_dict.items():
            for s_name, s_obj in m_obj.strat.items():
                # Verifies if Strat has WFM Config
                if not hasattr(s_obj, 'operation') or not isinstance(s_obj.operation, Walkforward):
                    continue
                    
                for a_name in m_obj.assets:
                    print(f"      \n>>> Running Walkforward Analisys: {m_name} | {s_name} | {a_name}")

                    # Recovers param_set matrix from cache
                    asset_node = self._results_map[self.name]["models"][m_name]["strats"][s_name]["assets"][a_name]
                    wfm_matrix = asset_node.get("wfm_matrix_data")

                    if wfm_matrix is None or wfm_matrix.is_empty():
                        print(f"      > [Skip] No WFM found for {a_name}")
                        continue

                    # Configues Engine with current matrix
                    wfm_engine = s_obj.operation
                    wfm_engine.matrix = wfm_matrix

                    # Executes and returns returns_mode='selected' result combination
                    wf_final_results = wfm_engine.analyze()
                    wf_parset_metric = str(wfm_engine.wf_selection_metric)
                    res_key = 'total_pnl' if wf_parset_metric == 'pnl' else wf_parset_metric

                    if not wf_final_results:
                        print(f"      > [Error] Walkforward failed to generate results for {a_name}")
                        continue

                    # Visualization
                    if wfm_engine.wf_returns_mode == 'selected':
                        display_val = wf_final_results.get(res_key, 0.0) # Mode TOP wf_final_results is already already best 
                        asset_node['walkforward'] = {
                            "best_result": wf_final_results,
                            "mode": "selected",
                            "metric_wf_parset": wf_parset_metric
                        }
                    else: # Mode ALL, wf_final_results is a dict with dicts
                        def get_val(key):
                            res = wf_final_results.get(key, {})
                            v = res.get(res_key, 0.0) if isinstance(res, dict) else 0.0
                            if hasattr(v, "item"): return float(v.item())
                            try: return float(v)
                            except: return 0.0
                        # def get_val(val):
                        #     val = wf_final_results.get(res_key, 0.0)
                        #     if hasattr(val, "item"): return float(val.item())
                        #     try: return float(val)
                        #     except: return 0.0
                            
                        best_key = builtins.max(wf_final_results.keys(), key=get_val)
                        display_val = get_val(best_key)
                        asset_node["walkforward"] = {
                            "all_results": wf_final_results,
                            "mode": "all"
                        }
                    
                    print(f"   > Walkforward Complete. Best {wf_parset_metric.upper()}: {display_val:.8f}")

                    if wf_final_results:
                        wfm_engine.plot_oos_curves(wf_result=wf_final_results)
                        wfm_engine.plot_timeline(wf_result=wf_final_results)
                        wfm_engine.plot_advanced_heatmap(metric=res_key)


        return True

    # || ======================================================================================================================================================================= ||
                        
    def run(self):
        # I - Init and Validation of Operation
        print(f"\n>>> I - Init and Validating Operation <<<")
        self._validate_operation()

        # II - Data Pre-Processing and Execution
        print(f"\n>>> II - Data Pre-Processing, Calculating Param Sets, Indicators, Signals and Backtests <<<")
        
        import time
        t0 = time.perf_counter()

        self._operation()
        #self._report_pnl_summary()
        #self._plot_pnl_curves()
        t1 = time.perf_counter()
        print(f"   > [PERF] in operation: {t1-t0:.3f}s")
        self._run_walkforward()

        # III - Operation Portfolio Simulation, Operation Analysis and Metrics
        print(f"\n>>> III - Operation Portfolio Simulation, Operation Analysis and Metrics <<<")
        self._simulate_portfolio()

        # IV - Pos-Processing, Saving and Cleaning
        print(f"\n>>> IV - Pos-Processing, Saving and Cleaning <<<")
        self._save_and_clean()

        return self._results_map

# || ======================================================================================================================================================================= ||



if __name__ == "__main__":
    eurusd = Asset(
        name='EURUSD',
        type='currency_pair',
        market='forex',
        data_path=f'C:\\Users\\Patrick\\Desktop\\Artaxes Portfolio\\MAIN\\MT5_Dados\\Forex')
    gbpusd = Asset(
        name='GBPUSD',
        type='currency_pair',
        market='forex',
        data_path=f'C:\\Users\\Patrick\\Desktop\\Artaxes Portfolio\\MAIN\\MT5_Dados\\Forex')
    usdjpy = Asset(
        name='USDJPY',
        type='currency_pair',
        market='forex',
        data_path=f'C:\\Users\\Patrick\\Desktop\\Artaxes Portfolio\\MAIN\\MT5_Dados\\Forex')
    winfut = Asset(
        name='WIN$',
        type='futures',
        market='b3',
        data_path=f'C:\\Users\\Patrick\\Desktop\\Artaxes Portfolio\\MAIN\\MT5_Dados')

    global_assets = {'EURUSD': eurusd, 'GBPUSD': gbpusd, 'USDJPY': usdjpy, 'WIN$': winfut} # Global Assets, loaded when app starts up, has all Asset and Portfolios 

    # =======================================================================================================|| Global Above

    model_assets=['EURUSD'] # Only keys #, 'GBPUSD'
    model_execution_tf = 'M15'

    strat_param_sets = {
        'AT15': { 
            'execution_tf': model_execution_tf,
            'backtest_start_idx': 21,
            'limit_order_exclusion_after_period': 1,
            'limit_order_perc_treshold_for_order_diff': 0.03,
            'limit_can_enter_at_market_if_gap': False,
            'limit_opposite_order_closes_pending': False,

            'exit_nb_only_if_pnl_is': 0, 
            'exit_nb_long': range(3, 10+1, 7),
            'exit_nb_short': range(3, 10+1, 7),
            
            'sl_perc': range(2, 10+1, 4), # 3
            'tp_perc': range(2, 10+1, 4), 
            'rr': range(2, 2+1, 2), 
            'param1': range(21, 21+1, 21), #50
            'param2': range(8, 24+1, 8), # 3
            'param3': ['sma'] #, 'ema', 'ema'
        }
    }

    from MA import MA # type: ignore
    from ATR_SL import ATR_SL # type: ignore
    from RawData import RawData # type: ignore
    from PriorCote import PriorCote # type: ignore
    from DayOpen import DayOpen # type: ignore

    # User imput Indicators
    ind = { 
        'atr': ATR_SL(asset=None, timeframe=model_execution_tf, window='param2'),
        #'ema': MA(asset=None, timeframe=model_execution_tf, window='param1', ma_type='param3', price_col='close'),
        #'ma': MA(asset='USDJPY', timeframe='D1', window='param1', ma_type='param3', price_col='close'),
        # 'htf_ma': MA(asset=None, timeframe='H1', window='param1', ma_type='param3', price_col='close'),
        # 'max': PriorCote(asset=None, timeframe=model_execution_tf, price_col='high'),
        # 'min': PriorCote(asset=None, timeframe=model_execution_tf, price_col='low'),
        # 'open_day': DayOpen(assertsset=None, timeframe=model_execution_tf),
    }

    def strat_signals(df: pl.DataFrame, params: dict) -> dict:
        # Can use columns df['high'] or str 'high' to point

        atr = df['atr']
        #ema = df['ema']

        bull = df['close'] < df['open']
        bear = df['close'] > df['open']

        entry_long  = bull & bull.shift(1) & bull.shift(2) #& (ema < ema.shift(1))
        entry_short = bear & bear.shift(1) & bear.shift(2) #& (ema > ema.shift(1))

        exit_tf_long  = bear & bear.shift(1)
        exit_tf_short = bull & bull.shift(1)

        # Preço da ordem pendente
        limit_long_price  = df['open'] #'high' #'high[1]' #
        limit_short_price = df['open'] #'low' #'low[1]' #

        # Distâncias (definidas ANTES de serem usadas)
        sl_long_price  = limit_long_price - atr * params['sl_perc'] 
        sl_short_price = limit_long_price + atr * params['sl_perc'] 

        # TP absoluto: 2R
        tp_long_price  = limit_long_price + atr  * params['tp_perc']
        tp_short_price = limit_long_price - atr * params['tp_perc']

        # Trailing: 0.5R
        trail_long_dist  = None #sl_long_dist  * 0.5
        trail_short_dist = None #sl_short_dist * 0.5

        # BE: distância de 1R para ativar (C++ faz: price >= entry + be_dist)
        be_long_dist  = None #sl_long_dist  * 1.0
        be_short_dist = None #sl_short_dist * 1.0

        
        if entry_long is not None and not isinstance(entry_long, str): entry_long = entry_long.shift(1)
        if entry_short is not None and not isinstance(entry_short, str): entry_short = entry_short.shift(1)
        if exit_tf_long is not None and not isinstance(exit_tf_long, str): exit_tf_long = exit_tf_long.shift(1)
        if exit_tf_short is not None and not isinstance(exit_tf_short, str): exit_tf_short = exit_tf_short.shift(1)
        
        if tp_long_price is not None and not isinstance(tp_long_price, str): tp_long_price = tp_long_price.shift(1)
        if tp_short_price is not None and not isinstance(tp_short_price, str): tp_short_price = tp_short_price.shift(1)
        if sl_long_price is not None and not isinstance(sl_long_price, str): sl_long_price = sl_long_price.shift(1)
        if sl_short_price is not None and not isinstance(sl_short_price, str): sl_short_price = sl_short_price.shift(1)
        
        if limit_long_price is not None and not isinstance(limit_long_price, str): limit_long_price = limit_long_price.shift(1)
        if limit_short_price is not None and not isinstance(limit_short_price, str): limit_short_price = limit_short_price.shift(1)
        
        if trail_long_dist is not None and not isinstance(trail_long_dist, str): trail_long_dist = trail_long_dist.shift(1)
        if trail_short_dist is not None and not isinstance(trail_short_dist, str): trail_short_dist = trail_short_dist.shift(1)
        
        if be_long_dist is not None and not isinstance(be_long_dist, str): be_long_dist = be_long_dist.shift(1)
        if be_short_dist is not None and not isinstance(be_short_dist, str): be_short_dist = be_short_dist.shift(1)
        return {
            'entry_long':       entry_long,
            'entry_short':      entry_short,
            'exit_long':        exit_tf_long,
            'exit_short':       exit_tf_short,

            'sl_price_long':    sl_long_price,
            'sl_price_short':   sl_short_price,
            'tp_price_long':    tp_long_price,
            'tp_price_short':   tp_short_price,

            'limit_long':       limit_long_price,
            'limit_short':      limit_short_price,

            'trail_long':       trail_long_dist,
            'trail_short':      trail_short_dist,

            'be_trigger_long':  be_long_dist,
            'be_trigger_short': be_short_dist,

            '__sig_key_params': ['param2', 'sl_perc', 'tp_perc']
        }


    # XXX - Recriar ponte py - cpp - py
    # XXX - Recriar sistema de regras para ficar mais simples (py gera sinal - cpp executa)
    # XXX - Optimization 

    # XXX - Existe um problema no updated pnl daily, se eu considero um novo parset com trade comprado ainda vou estar simulando a variação baseada na abertura, como tratar? \
    #talvez colocar que se trocou o parset e o parset novo já tem trade aberto ele considera a variação do pct_change e não do (close-open)/open, logo qualquer nova variação negativa -, positiva +
    # XXX - Adicionar lado, WFM que pode selecionar optmizize LONG, SHORT or BOTH sides tanto em WFM quanto Portfolio Simulator. Redundante salvar lado/asset/model/strat, se orientar pelo _results_map
    # - Desenvolver MM sistema de slippage, lot, comission, etc; Tanto em py tanto cpp, Model lida com Asset
    # - Multi entry and lot strategy

    # - Dev Roadmap png/list 
    # - Organize _results_map with ps_id and param_set_key
    # - Develop start_date - end_date for operation

    # - Plot long list with small leters with selectable mode-strat-asset-parest/wf results
    # - List above should show parset ps_id and param_set_key for all OS wfm results

    # - Criar base de dados SQL com sistema de batch para armazenar e gerenciar dados
    # - Realocar dados e settings do Assets para SQL?
    
    # - Adicionar Backtest M1 (procura converter sinais para M1 se dado disponível)
    # - Adicionar novo Backtester para Close-Close, Open-Open, Tick. Vetoriazado e não vetorizado [i]

    # - Deselop SM selection system for Models/Strats/Assets
    # - Develop Portfolio Simulator

    # PortfolioSimulator deve ter a opção de ter uma matrix de covariancia para models uma para strats e uma para assets? talvez uma que armazene as posições selecionadas apenas?

    AT15 = Strat(
        StratParams(
            name="AT15",
            operation=Walkforward(
                wfm_configs=[[is_len, os_len, os_len] for is_len, os_len in itertools.product([1, 2, 4, 12, 16, 24, 36, 48], [1, 2, 4, 12, 16, 24, 36, 48])],
                wfm_is_always_higher_or_equal_to_oos=True,
                matrix_resolution='weekly', time_mode = 'calendar_days',
                is_metric='pnl', is_top_n=1, is_logic='highest', is_order='des',
                wf_selection_metric='wfe', wf_selection_analysis_radius_n=1,
                wf_selection_logic='highest_stable', wf_returns_mode='selected'
            ),
            execution_settings=ExecutionSettings(hedge=True, strat_num_pos=[1,1], strat_max_num_pos_per_day=[999,999],
                                                 order_type='market', limit_order_base_calc_ref_price='open', 
                                                 slippage=0.0, comission=0.0, 
                                                 day_trade=False, timeTI=None, timeEF=None, timeTF=None, next_index_day_close=False, # "0:00"
                                                 day_of_week_close_and_stop_trade=[], timeExcludeHours=None, dateExcludeTradingDays=None, dateExcludeMonths=None, 
                                                 fill_method='ffill', fillna=0, trade_pnl_resolution='daily', 
                                                 print_logs=False),
            mma_settings=None, # If mma_rules=None then will use default or PMA or other saved MMA define in Operation. Else it creates a temporary MMA with mma_settings
            params=strat_param_sets['AT15'], # SE signal_params então iterar apenas nos parametros do signal_params para criar sets, else usa apenas sets do indicadores, else sem sets
            indicators=ind,
            signals=strat_signals
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
            assets=global_assets,
            operation_timeframe=model_execution_tf, # Must always be the smaller timeframe among all strat execution_timeframe
            date_start=None, #'2020-01-01',
            date_end=None, #'2023-01-01',
            save=False,
            metrics={}
        )
    )

    operation.run()










