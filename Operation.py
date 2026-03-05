from webbrowser import get

import polars as pl, numpy as np, json, sys, uuid, copy, datetime, psutil, re, itertools
sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend\Indicators')
sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend')

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

    def _translate_signals(self, signal_list):
        """ Converte a lista de objetos gerados pela classe Col em JSON puro. """
        if not signal_list: return []
        # Garante que estamos lidando com uma lista de dicts (as regras)
        return [rule if isinstance(rule, dict) else rule for rule in signal_list]

    def _operation(self):
        models = self._get_all_models()
        self._results_map[self.name] = {'models': {}}
        unique_indicators_lists = {}

        for model_name, model_obj in models.items():
            strats = model_obj.strat
            assets = model_obj.assets
            model_tf = model_obj.execution_timeframe
            self._results_map[self.name]['models'][model_name] = {'strats': {}}

            for strat_name, strat_obj in strats.items():
                param_sets = self._calculate_param_combinations(strat_obj.params)
                self._results_map[self.name]['models'][model_name]['strats'][strat_name] = {'assets': {}}

                for asset_name in assets:
                    asset_class = self.assets.get(asset_name)
                    if not asset_class: continue
                    self._results_map[self.name]['models'][model_name]['strats'][strat_name]['assets'][asset_name] = {'param_sets': {}}
                    base_asset_df = asset_class.data_get(model_tf)

                    # Converts to dict
                    exec_set_raw = asdict(strat_obj.execution_settings)
                    
                    # Normalize for cpp
                    exec_set_mod = self.prepare_time_params(exec_set_raw)
        
                    # Creates batch preset
                    asset_batch = {
                        "asset_header": f"{model_name}_{strat_name}_{asset_name}",
                        "data": base_asset_df.to_dict(as_series=False),
                        "execution_settings": exec_set_mod,
                        "simulations": []
                    }

                    for ps_name, ps_dict in param_sets.items():
                        self._results_map[self.name]['models'][model_name]['strats'][strat_name]['assets'][asset_name]['param_sets'][ps_name] = {
                            'param_set_dict': ps_dict,
                            'trades': []
                        }
                        indicator_data = {}

                        for ind_key, ind_obj in strat_obj.indicators.items():
                            eff_params = self.effective_params_from_global(ind_obj.params, ps_dict)
                            ind_p_hash = self.param_suffix(eff_params)
                            unique_key = f"{asset_name}_{model_tf}_{ind_key}_{ind_p_hash}"

 

                           # 1. OK? Estava com nome do ind errado, agora como estou passando para o obj ele deveria funcionar
                           # 2. Não está calculando ou traduzindo HTF->LTF ind ou calculate?, NÃO CONFUNDIR HTF-LTF COM PRIORCOTE





                            if unique_key not in unique_indicators_lists:
                                print(f"   > Calculating indicator >>> name: {ind_key} >>> unique_key: {unique_key}")

                                temp_ind_df = self._calculate_indicator(
                                    model_timeframe=model_tf,
                                    ind_name=ind_key,
                                    ind_obj=ind_obj,
                                    param_set_dict=ps_dict,
                                    curr_asset_obj=base_asset_df, 
                                    asset_name=asset_name,
                                    datetime_candle_references=asset_class.datetime_candle_references
                                )

                                novas_colunas = [c for c in temp_ind_df.columns if c not in base_asset_df.columns]

                                col_data_map = {}
                                for col in novas_colunas:
                                    col_data_map[col] = temp_ind_df[col].to_list()

                                unique_indicators_lists[unique_key] = col_data_map
                            else:
                                print(f"   > Using cache indicator >>> name: {ind_key} >>> unique_key: {unique_key}")

                            indicator_data.update(unique_indicators_lists[unique_key])

                        # debug_df = base_asset_df.clone()
                        # for col_name, values_list in indicator_data.items(): debug_df = debug_df.with_columns(pl.Series(name=col_name, values=values_list))
                        # output_filename = f"debug_{asset_name}_{ps_name}.xlsx"
                        # debug_df.tail(-500).write_excel(output_filename)

                        asset_batch["simulations"].append({
                            "id": ps_name,
                            "params": ps_dict,
                            "indicator_data": indicator_data,
                            "rules": {
                                "entry_long": self._translate_signals(strat_obj.signal_rules.get('entry_long')),
                                "entry_short": self._translate_signals(strat_obj.signal_rules.get('entry_short')),

                                "entry_long_limit_position": self._translate_signals(strat_obj.signal_rules.get('entry_long_limit_position')),
                                "entry_short_limit_position": self._translate_signals(strat_obj.signal_rules.get('entry_short_limit_position')),
                                "entry_long_limit_value": self._translate_signals(strat_obj.signal_rules.get('entry_long_limit_value')),
                                "entry_short_limit_value": self._translate_signals(strat_obj.signal_rules.get('entry_short_limit_value')),

                                "exit_tf_long": self._translate_signals(strat_obj.signal_rules.get('exit_tf_long')),
                                "exit_tf_short": self._translate_signals(strat_obj.signal_rules.get('exit_tf_short')),

                                "exit_sl_long_price": self._translate_signals(strat_obj.signal_rules.get('exit_sl_long_price')),
                                "exit_tp_long_price": self._translate_signals(strat_obj.signal_rules.get('exit_tp_long_price')),
                                "exit_sl_short_price": self._translate_signals(strat_obj.signal_rules.get('exit_sl_short_price')),
                                "exit_tp_short_price": self._translate_signals(strat_obj.signal_rules.get('exit_tp_short_price')),

                                'be_pos_long_signal': self._translate_signals(strat_obj.signal_rules.get('be_pos_long_signal')),
                                'be_pos_short_signal': self._translate_signals(strat_obj.signal_rules.get('be_pos_short_signal')),
                                'be_neg_long_signal': self._translate_signals(strat_obj.signal_rules.get('be_neg_long_signal')),
                                'be_neg_short_signal': self._translate_signals(strat_obj.signal_rules.get('be_neg_short_signal')),

                                'be_pos_long_value': self._translate_signals(strat_obj.signal_rules.get('be_pos_long_value')),
                                'be_pos_short_value': self._translate_signals(strat_obj.signal_rules.get('be_pos_short_value')),
                                'be_neg_long_value': self._translate_signals(strat_obj.signal_rules.get('be_neg_long_value')),
                                'be_neg_short_value': self._translate_signals(strat_obj.signal_rules.get('be_neg_short_value')),
                            }
                        })

                    # 4. Envio único para C++
                    print(f"   > Processing {asset_name}: {len(param_sets)} simulations...")
                    full_output = self._run_cpp_operation(asset_batch)
                    self._save_trades(full_output, model_name, strat_name, asset_name)

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

    def _operation_analysis(self):
        pass

    # || ===================================================================== || Execution Functions || ===================================================================== ||

    def _run_cpp_operation(self, asset_batch: dict):
        try:
            # Configuração do Caminho da DLL
            path_to_dll = r"C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend\build\Release"
            if path_to_dll not in sys.path: 
                sys.path.append(path_to_dll)
            
            import engine_cpp # type: ignore

            # 1. PREPARAÇÃO DO DATAFRAME (Convertemos aqui para garantir tipos corretos)
            # Se o 'data' no asset_batch já for um dict vindo do to_dict(), transformamos em DF para tratar
            if isinstance(asset_batch['data'], dict):
                df = pl.DataFrame(asset_batch['data'])
            else:
                df = asset_batch['data'].clone()

            # Mantemos apenas datetime + colunas numéricas (Float ou Int)
            # Isso remove 'ativo', 'date' e 'time' que causaram o erro 302
            df = df.select([
                pl.col("datetime"),
                pl.col(pl.Float64),
                pl.col(pl.Int64),
                pl.col(pl.Float32)
            ])
                
            # --- TRATAMENTO DE DATETIME ---
            if 'datetime' in df.columns and df.schema['datetime'] != pl.Utf8:
                df = df.with_columns(pl.col('datetime').dt.to_string('%Y-%m-%d %H:%M:%S'))

                # if numeric_cols:
                #     df = df.with_columns([
                #         pl.col(c).cast(pl.Float64).fill_nan(0.0).fill_null(0.0) 
                #         for c in numeric_cols
                #     ])

            # --- TRATAMENTO NUMÉRICO (Cast para Float64) ---
            numeric_cols = [name for name, dtype in df.schema.items() if dtype.is_numeric()]
            if numeric_cols:
                df = df.with_columns([pl.col(c).cast(pl.Float64).fill_null(0.0) for c in numeric_cols])

            # 2. MONTAGEM DO PAYLOAD FINAL (Nomes de chaves alinhados com Operation::run no C++)
            final_payload = {
                "asset_header": asset_batch.get('asset_header', 'Unknown'),
                "data": df.to_dict(as_series=False), # Agora só tem números e a string datetime
                "time_settings": asset_batch.get('time_settings', {}),
                "execution_settings": asset_batch.get('execution_settings', {}),
                "simulations": asset_batch.get('simulations', [])
            }

            # 3. SERIALIZAÇÃO SEGURA
            def json_serial(obj):
                """Lida com objetos que o json standard não consegue serializar."""
                if isinstance(obj, (datetime.datetime, datetime.date)):
                    return obj.isoformat()
                # Se for um objeto de Regra (Col/Params), tenta pegar o dict dele
                if hasattr(obj, 'to_dict'):
                    return obj.to_dict()
                return str(obj)

            # Transforma tudo em uma string JSON para o C++
            json_str = json.dumps(final_payload, default=json_serial)
            
            # 4. CHAMADA DA ENGINE C++
            # O engine_cpp.run() agora retorna uma STRING (json.dump do C++)
            raw_output = engine_cpp.run(json_str)
            
            # 5. PARSE DO RESULTADO
            if not raw_output or raw_output == "[]" or raw_output == "{}":
                return {"simulations": [], "wfm_data": []}
                
            # Como o C++ retornou uma string, precisamos de json.loads para voltar a ser lista/dict
            if isinstance(raw_output, str):
                parsed = json.loads(raw_output)
                return {
                    "simulations": parsed.get("simulations", []),
                    "wfm_data": parsed.get("wfm_data", [])
                }
            
            return raw_output

        except Exception as e:
            print(f'< Error in Python-C++ Bridge: {e}')
            return {"simulations": [], "wfm_data": []}

    def _save_trades(self, full_output: dict, m_name: str, s_name: str, a_name: str):
        if not full_output: return

        simulations = full_output.get("simulations", [])
        wfm_raw = full_output.get("wfm_data", [])

        # Process Trades
        for simulation_batch in simulations:
            for trade in simulation_batch:
                full_key = trade.get("path", "")
                idx_param = full_key.find("_param_set")
                
                if idx_param == -1:
                    print(f"DEBUG: Formato de path inesperado: {full_key}")
                    continue
                ps_name = full_key[idx_param+1:] # "param_set-21-..."

                try: 
                    target = self._results_map[self.name]["models"][m_name]["strats"][s_name]["assets"][a_name]["param_sets"][ps_name]
                    if target["trades"] is None: target["trades"] = []
                    trade["asset"] = a_name 
                    target["trades"].append(trade)
                    
                except KeyError as e:
                    print(f"DEBUG: Chave não encontrada: {e} | Caminho tentado: {m_name}->{s_name}->{a_name}->{ps_name}")

        # Process WFM Daily Results
        if wfm_raw:
            try:
                # Transforms cpp dict lists to Polars DataFrame pivots
                df_matrix = self._process_wfm_to_polars(wfm_raw)
                map_asset_nome = self._results_map[self.name]["models"][m_name]["strats"][s_name]["assets"][a_name]
                map_asset_nome["wfm_matrix_data"] = df_matrix
                print(f"   > WFM Matrix generated for {m_name} | {s_name} | {a_name}: {df_matrix.shape}")
            except Exception as e:
                print(f"   > Error generating WFM Matrix for {m_name} | {s_name} | {a_name}: {e}")

    def _process_wfm_to_polars(self, wfm_raw):
        df = pl.DataFrame(wfm_raw)

        # Converts saved timestamp YYYYMMDDHHMMSS to Datetime
        df = df.with_columns(
            pl.col("ts").cast(pl.Utf8).str.to_datetime("%Y%m%d%H%M%S")
        )

        # Pivot with SUM agg to avoid duplicate error
        matrix = df.pivot(
            on="id",
            index="ts",
            values="pnl",
            aggregate_function="sum"
        ).sort("ts").fill_null(0.0)
        
        # Rename columns to ps_0, ps_1, ..., ps_{col}.
        matrix = matrix.rename({
            col: f"ps_{col}" for col in matrix.columns if col != "ts"
        })

        return matrix

    # Batch System Defs
    def _estimate_paramset_size_mb(self, df: pl.DataFrame):
        return df.estimated_size() / (1024 ** 2) # No Polars, estimated_size() retorna o tamanho em bytes
    
    def _get_available_memory_mb(self):
        return psutil.virtual_memory().available / (1024 ** 2)
    
    def _calculate_optimal_batch_size(self, avg_paramset_size_mb, safety_margin=0.97, max_batch=1000, min_batch=1):
        available_ram_mb = self._get_available_memory_mb()
        usable_ram_mb = available_ram_mb * (1 - safety_margin)

        if avg_paramset_size_mb <= 0: return min_batch

        z = int(usable_ram_mb // avg_paramset_size_mb)
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
    

    """
    def transfer_htf_columns(self, ltf_df, htf_df, ind_name):
        
        # Alinha dados de ativos diferentes ou timeframes diferentes (HTF -> LTF).
        # Regra de Leakage:
        # - Se for 'open': Usa o valor do próprio candle (Safe: o Open não muda após a abertura).
        # - Se for outro (close, max, min): Garante o uso do dado do período anterior para 
        # evitar que o LTF conheça o fechamento/máxima/mínima antes do tempo.
        
        # 1. Limpeza e Ordenação (Essencial para o join_asof)
        ltf_df = ltf_df.sort("datetime")
        htf_df = htf_df.select(["datetime", ind_name]).sort("datetime")

        # 2. Tratamento de Leakage para dados que não são 'Open'
        # Se estamos alinhando o 'close' do D1 no M15, o M15 às 10:00 não pode ver 
        # o 'close' do dia que só existirá às 18:00.
        if ind_name.lower() != "open":
            # Deslocamos os dados HTF em 1 posição. Assim, o 'backward' join 
            # pegará o valor que estava disponível no fechamento do candle anterior.
            htf_df = htf_df.with_columns(
                pl.col(ind_name).shift(1)
            )

        # 3. Join Asof (Sincronização de Relógio)
        # 'backward' busca o último valor HTF onde HTF_datetime <= LTF_datetime
        aligned = ltf_df.join_asof(
            htf_df,
            on="datetime",
            strategy="backward"
        )

        # 4. Cálculo de Integridade
        total = len(aligned)
        nulos = aligned[ind_name].null_count()
        success_rate = ((total - nulos) / total) * 100 if total > 0 else 0
        
        # 5. Tratamento de Gaps e Início de Histórico
        # forward: preenche feriados/gaps com o último valor conhecido
        # backward: preenche o início do histórico (antes do HTF começar) com o primeiro valor disponível
        aligned = aligned.with_columns(
            pl.col(ind_name)
            .fill_null(strategy="forward")
            .fill_null(strategy="backward")
            .fill_null(0.0)
        )

        # Logging para monitoramento
        status = "OK" if success_rate > 85 else "⚠️ LOW COVERAGE"
        print(f"      > Aligned {ind_name}: {success_rate:.2f}% [{status}]")

        return aligned
    """
    
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

    def _save_and_clean(self):
        pass

    # || ===================================================================== || Metrics Functions || ===================================================================== ||

    def _report_pnl_summary(self):
        import builtins

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
                
                for asset_path, asset_data in strat_data.get("assets", {}).items():
                    # EXTRAÇÃO: Pegamos apenas o final do path para exibição
                    # Se asset_path for "MA Trend Following_AT15_EURUSD", display_name será "EURUSD"
                    display_name = asset_path.split('_')[-1]
                    
                    print(f"      └── Asset: {display_name}")
                    
                    for param_key, param_data in asset_data.get("param_sets", {}).items():
                        trades = param_data.get("trades", [])
                        
                        if not trades:
                            print(f"          └── {param_key}: No trades.")
                            continue

                        # Função auxiliar para extrair valores (suporta dict ou objeto)
                        def get_val(t, attr):
                            return t.get(attr, 0) if isinstance(t, dict) else getattr(t, attr, 0)



                        # for t in trades:
                        #     # Garante que estamos pegando as listas
                        #     pnls = t.get('daily_pnl', [])
                        #     dts = t.get('daily_datetime', [])
                        #     trade_id = t.get('id', 'N/A')

                        #     print(f"\n--- Histórico Diário do Trade {trade_id} ---")
                        #     if not pnls:
                        #         print("Nenhum dado diário capturado.")
                        #         continue

                        #     for pnl, dt in zip(pnls, dts):
                        #         # O C++ envia o PnL acumulado ou do dia; aqui printamos conforme recebido
                        #         print(f"Data: {dt} | PnL: {float(pnl):.4f}%")




                        # SEPARAÇÃO POR LADO (Baseado no sinal do lot_size)
                        longs = [t for t in trades if get_val(t, 'lot_size') > 0]
                        shorts = [t for t in trades if get_val(t, 'lot_size') < 0]

                        def calc_metrics(trade_list):
                            if not trade_list: 
                                return {"pnl": 0.0, "wr": 0.0, "cnt": 0, "avg": 0.0}
                            
                            #for t in trade_list: print(t)

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
                        if all_pnls:
                            best = builtins.max(all_pnls)
                            worst = builtins.min(all_pnls)
                        print(f"              {'-'*80}")
                        print(f"              Best Trade: {best:.2f}%  |  Worst Trade: {worst:.2f}%\n")

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
        self._operation()
        #self._report_pnl_summary()
        #self._plot_pnl_curves()
        
        self._run_walkforward()

        # III - Operation Portfolio Simulation, Operation Analysis and Metrics
        print(f"\n>>> III - Operation Portfolio Simulation, Operation Analysis and Metrics <<<")
        self._simulate_portfolio()

        # IV - Pos-Processing, Saving and Cleaning
        print(f"\n>>> IV - Pos-Processing, Saving and Cleaning <<<")
        self._save_and_clean()

        return self._results_map

# || ======================================================================================================================================================================= ||

class Col:
    def __init__(self, name, shift=0):
        self.name = name
        self.shift = shift

    # Allows using Col("close")[1] for shift 1
    def __getitem__(self, s):
        return Col(self.name, s)
    
    # Comparison operators
    def __gt__(self, other): return self._build_rule(other, ">")
    def __lt__(self, other): return self._build_rule(other, "<")
    def __ge__(self, other): return self._build_rule(other, ">=")
    def __le__(self, other): return self._build_rule(other, "<=")
    def __eq__(self, other): return self._build_rule(other, "==")

    def _build_rule(self, other, op):
        return {
            "a": self.name, 
            "shift_a": self.shift,
            "op": op,
            "b": other.name if isinstance(other, Col) else ("const" if not hasattr(other, 'to_dict') else "expr"),
            "shift_b": other.shift if isinstance(other, Col) else 0,
            "val": other if not (isinstance(other, Col) or hasattr(other, 'to_dict')) else None,
            "expr": other.to_dict() if hasattr(other, 'to_dict') else None
        }
    
    # Arithmetic operators
    def __sub__(self, other):
        return Expression(self, "=", other)
    def __add__(self, other):
        return Expression(self, "+", other)
    def __mul__(self, other):
        return Expression(self, "*", other)
    def rolling_mean(self, window):
        return Expression(self, "rolling_mean", window)
    def to_dict(self):
        return {"type": "col", "name": self.name, "shift": self.shift}
    
class Expression:
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def __mul__(self, other):
        return Expression(self, "*", other)
    def rolling_mean(self, window):
        return Expression(self, "rolling_mean", window)
    def to_dict(self):
        return {
            "type": "operation",
            "left": self.left.to_dict() if hasattr(self.left, 'to_dict') else self.left,
            "op": self.op,
            "right": self.right.to_dict() if hasattr(self.right, 'to_dict') else self.right
        }

class ParamRef:
    def __init__(self, name):
        self.name = name
    
    def __repr__(self):
        return f"PARAM_{self.name}"
    
    def to_dict(self):
        return {"type": "param", "name": self.name}
    
def Params(name):
    return ParamRef(name)

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
    winfut = Asset(
        name='WIN$',
        type='futures',
        market='b3',
        data_path=f'C:\\Users\\Patrick\\Desktop\\Artaxes Portfolio\\MAIN\\MT5_Dados'
    )

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
            'exit_nb_long': range(0, 0+1, 3),
            'exit_nb_short': range(0, 0+1, 3),
            
            'sl_perc': range(2, 8+1, 3), # 3
            'tp_perc': range(5, 5+1, 8), 
            'param1': range(21, 85+1, 21), #50
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
        'ema': MA(asset=None, timeframe=model_execution_tf, window='param1', ma_type='param3', price_col='close'),
        #'ma': MA(asset='USDJPY', timeframe='D1', window='param1', ma_type='param3', price_col='close'),
        # 'htf_ma': MA(asset=None, timeframe='H1', window='param1', ma_type='param3', price_col='close'),
        # 'max': PriorCote(asset=None, timeframe=model_execution_tf, price_col='high'),
        # 'min': PriorCote(asset=None, timeframe=model_execution_tf, price_col='low'),
        # 'open_day': DayOpen(assertsset=None, timeframe=model_execution_tf),
    }

    close = Col("close")
    open = Col("open")
    high = Col("high")
    low = Col("low")
    atr = Col("atr")
    ema = Col("ema")
    #ma = Col("ma")
    tp_perc = Params("tp_perc")
    sl_perc = Params("sl_perc")
    #htf_ma = Col("htf_ma")
    max = Col('max')
    min = Col('min')
    open_day = Col('open_day')



    # REIMPLEMENTAR SISTEMA DE SINAIS, GERAR SINAIS PY E ENVIAR SINAIS, COLUNAS RELEVANTES E 1 DF DE CADA ASSET RELEVANTE PARA CPP
    # def entry_long(self, df, params):
    #     # Regras de entrada (3 candles)
    #     df = df.with_columns(
    #         sig_entry_long = (pl.col("close") > pl.col("open")).rolling_sum(3) == 3
    #     )
    #     # Regra de saída (Cruzamento de média)
    #     ma = df["close"].ewm_mean(span=params['period'])
    #     df = df.with_columns(
    #         sig_exit_long = pl.col("close") < ma
    #     )
    #     return df


    entry_long = [
        close[1] < open[1],
        close[2] < open[2],
        close[3] < open[3],
        close[1] < ema[1],
        ema[1] < ema[2],
    ]
    entry_short = [
        close[1] > open[1],
        close[2] > open[2],
        close[3] > open[3],
        close[1] > ema[1],
        ema[1] > ema[2],
    ]

    exit_tf_long = [
        close[1] > open[1],
        close[2] > open[2],
    ]
    exit_tf_short = [
        close[1] < open[1],
        close[2] < open[2],
    ]

    exit_tp_long_price = None #[2000] #[atr * tp_perc]
    exit_tp_short_price = None #[2000] #[atr * tp_perc]

    exit_sl_long_price = [atr * sl_perc] #[400] #
    exit_sl_short_price = [atr * sl_perc] #[400] #

    entry_long_limit_position = None #["low"]
    entry_short_limit_position = None #["high"]
    entry_long_limit_value = None #[0.0002]
    entry_short_limit_value = None #[0.0002]

    be_pos_long_signal = None #[close > close[1]]
    be_pos_short_signal = None #[close < close[1]]
    be_neg_long_signal = None #[close < close[1]]
    be_neg_short_signal = None #[close > close[1]]

    be_pos_long_value = None #[atr]
    be_pos_short_value = None #[atr]
    be_neg_long_value = None #[atr]
    be_neg_short_value = None #[atr]

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
                                                 order_type='market', limit_order_base_calc_ref_price='open', offset=0.0, 
                                                 day_trade=False, timeTI=None, timeEF=None, timeTF=None, next_index_day_close=False, # "0:00"
                                                 day_of_week_close_and_stop_trade=[], timeExcludeHours=None, dateExcludeTradingDays=None, dateExcludeMonths=None, 
                                                 fill_method='ffill', fillna=0, trade_pnl_resolution='daily'),
            mma_settings=None, # If mma_rules=None then will use default or PMA or other saved MMA define in Operation. Else it creates a temporary MMA with mma_settings
            params=strat_param_sets['AT15'], # SE signal_params então iterar apenas nos parametros do signal_params para criar sets, else usa apenas sets do indicadores, else sem sets
            indicators=ind,
            signal_rules={
                'entry_long': entry_long,
                'entry_short': entry_short,

                'entry_long_limit_position': entry_long_limit_position,
                'entry_short_limit_position': entry_short_limit_position,
                'entry_long_limit_value': entry_long_limit_value,
                'entry_short_limit_value': entry_short_limit_value,

                'exit_tf_long': exit_tf_long,
                'exit_tf_short': exit_tf_short,

                'exit_sl_long_price': exit_sl_long_price,
                'exit_sl_short_price': exit_sl_short_price,
                'exit_tp_long_price': exit_tp_long_price,
                'exit_tp_short_price': exit_tp_short_price,

                'be_pos_long_signal': be_pos_long_signal,
                'be_pos_short_signal': be_pos_short_signal,
                'be_neg_long_signal': be_neg_long_signal,
                'be_neg_short_signal': be_neg_short_signal,

                'be_pos_long_value': be_pos_long_value,
                'be_pos_short_value': be_pos_short_value,
                'be_neg_long_value': be_neg_long_value,
                'be_neg_short_value': be_neg_short_value,
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
            assets=global_assets,
            operation_timeframe=model_execution_tf, # Must always be the smaller timeframe among all strat execution_timeframe
            date_start=None, #'2020-01-01',
            date_end=None, #'2023-01-01',
            save=False,
            metrics={}
        )
    )

    operation.run()



"""
- Implement all remaining Strat and Operation config minor methods
- Optimize performance, test new backtest method?

- Pente fino em cada método e operação
- Sistema de Simulação de Portfolio com suporte de Modelos para filtro/seleção de Asset, Strat e Backtest (param_set)
- Walkforward / WFM
- Modulo de Summary, Plot e Analise de Resultados
- Sistema de armazenamento de dados de Asset (substituir var local global_assets)
- Support for tick data


"""
"""
def _extract_returns_for_wf(self, model_name, strat_name, asset_name): # Extracts all param_sets daily historical returns for wf
    wf_input_data = []

    asset_data = self._results_map[self.name]["models"][model_name]["strats"][strat_name]["assets"][asset_name]
    param_sets = asset_data.get("param_sets", {})

    for ps_name, ps_data in param_sets.items():
        trades = ps_data.get("trades", [])
        if not trades: continue

        # Converts trades list into a dict
        ts_map = {}
        for t in trades:
            dts = t.get('daily_datetime', [])
            pnls = t.get('daily_pnl', [])

            for dt, pnl in zip(dts, pnls):
                ts_map[dt] = ts_map.get(dt, 0.0) + pnl

        if not ts_map: continue

        # Cronological order to garantee walkforard integrity
        sorted_dts = sorted(ts_map.keys())
        sorted_pnls = [ts_map[dt] for dt in sorted_dts]
        wf_input_data.append([sorted_dts, sorted_pnls, ps_name])
    return wf_input_data

def _run_walkforward(self): # Executes WFM for all results
    models_dict = self._get_all_models()

    for m_name, m_obj in models_dict.items():
        strats = m_obj.strat
        assets = m_obj.assets

        for s_name, s_obj in strats.items():

            # Checks strat configuration for if walkforward parameters
            if not hasattr(s_obj, 'operation') or not isinstance(s_obj.operation, Walkforward): continue

            # Iterates over assets to extract returns and run walkforward for each asset in model
            for a_name in assets:
                print(f"\n>>> Running Walkforward for Model: {m_name} | Strat: {s_name} | Asset: {a_name} <<<")
                
                wf_raw_data = self._extract_returns_for_wf(m_name, s_name, a_name)

                if not wf_raw_data:
                    print(f"< No data extracted for Walkforward in {m_name} | {s_name} | {a_name}. Skipping.")
                    continue

                engine_input = [[d[0], d[1]] for d in wf_raw_data]
                wf_engine = s_obj.operation

                # Updates internal engine dataframe with the extracted data, now in the wide format expected by the C++ engine
                wf_engine.df = wf_engine._consolidate_to_wide_df(engine_input)

                # Executes Walkforward Matrix and returns combined OOS curve from all param sets
                wf_final_results = wf_engine.analyze()

                # Saves consolidated results on asset level (model->strat->asset)
                self._results_map[self.name]["models"][m_name]["strats"][s_name]["assets"][a_name]["walkforward"] = {
                    "trades": wf_final_results,
                    "matrix": wf_engine.all_wf_results,
                    "config": wf_engine.wfm_configs
                }

                wf_engine.plot_advanced_heatmap(metric='total_pnl')


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
    


"""






