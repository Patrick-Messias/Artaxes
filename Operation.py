import polars as pl, numpy as np, json, sys, uuid, copy, datetime, psutil
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
from MA import MA # type: ignore
from ATR_SL import ATR_SL # type: ignore

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
                    
                    # 1. Puxamos o Dataframe único do Ativo com indicadores estáticos
                    base_asset_df = asset_class.data_get(model_tf)
                    
                    # 2. Criamos o Batch por Ativo
                    asset_batch = {
                        "asset_header": f"{model_name}_{strat_name}_{asset_name}",
                        "data": base_asset_df.to_dict(as_series=False),
                        "execution_settings": asdict(strat_obj.execution_settings),
                        "simulations": []
                    }

                    # 3. Adicionamos as simulações (Parâmetros + Regras de Sinais)
                    for ps_name, ps_dict in param_sets.items():
                        self._results_map[self.name]['models'][model_name]['strats'][strat_name]['assets'][asset_name]['param_sets'][ps_name] = {
                            'param_set_dict': ps_dict, 
                            'trades': None
                        }

                        # Calculates indicator
                        current_df = base_asset_df.clone()
                        for ind_key, ind_obj in strat_obj.indicators.items():
                            print(f"   > Calculating indicator: {ind_key}")
                            res = ind_obj.calculate(current_df, ps_dict).fill_null(0)
                            current_df = current_df.with_columns(
                                res.cast(pl.Float64).alias(ind_key)
                            )

                        new_cols = [ind_key for ind_key, _ in strat_obj.indicators.items()]
                        indicator_data = current_df.select(new_cols).to_dict(as_series=False)

                        asset_batch["simulations"].append({
                                "id": ps_name,
                                "params": ps_dict,
                                "indicator_data": indicator_data,
                                "rules": {
                                    "entry_long": self._translate_signals(strat_obj.signal_rules.get('entry_long')),
                                    "entry_short": self._translate_signals(strat_obj.signal_rules.get('entry_short')),
                                    "exit_sl_long_price": self._translate_signals(strat_obj.signal_rules.get('exit_sl_long_price')),
                                    "exit_tp_long_price": self._translate_signals(strat_obj.signal_rules.get('exit_tp_long_price')),
                                    "exit_sl_short_price": self._translate_signals(strat_obj.signal_rules.get('exit_sl_short_price')),
                                    "exit_tp_short_price": self._translate_signals(strat_obj.signal_rules.get('exit_tp_short_price')),
                                    "exit_tf_long": self._translate_signals(strat_obj.signal_rules.get('exit_tf_long')),
                                    "exit_tf_short": self._translate_signals(strat_obj.signal_rules.get('exit_tf_short'))
                                }
                            })

                    # 4. Envio único para C++
                    print(f"   > Processing {asset_name}: {len(param_sets)} simulations...")
                    trades = self._run_cpp_operation(asset_batch)
                    self._save_trades(trades)

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
            if not raw_output or raw_output == "[]":
                return []
                
            # Como o C++ retornou uma string, precisamos de json.loads para voltar a ser lista/dict
            if isinstance(raw_output, str):
                return json.loads(raw_output)
            
            return raw_output

        except Exception as e:
            print(f'< Error in Python-C++ Bridge: {e}')
            import traceback
            traceback.print_exc()
            return []

    def _save_trades(self, raw_output):
        if not raw_output: return

        for simulation_batch in raw_output:
            for trade in simulation_batch:
                full_key = trade.get("asset", "") # "MA Trend Following_AT15_EURUSD_param_set-21-..."
                
                # Identificamos onde começa o param_set
                idx_param = full_key.find("_param_set")
                if idx_param == -1: continue
                
                ps_name = full_key[idx_param+1:] # "param_set-21-..."
                prefix = full_key[:idx_param]    # "MA Trend Following_AT15_EURUSD"
                
                parts = prefix.split('_')
                # parts[0] = Model, parts[1] = Strat, parts[2] = Asset
                
                try:
                    # Caminho exato no mapa de resultados
                    target = self._results_map[self.name]["models"][parts[0]]["strats"][parts[1]]["assets"][parts[2]]["param_sets"][ps_name]
                    
                    if target["trades"] is None:
                        target["trades"] = []
                    
                    target["trades"].append(trade)
                except KeyError as e:
                    print(f"DEBUG: Chave não encontrada no results_map: {e} | PS tentado: {ps_name}")

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
        indicator_res = ind_obj.calculate(df_source, param_set_dict=param_set_dict)
        indicator_result = indicator_res.fill_null(0)

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

    global_assets = {'EURUSD': eurusd, 'GBPUSD': gbpusd, 'USDJPY': usdjpy} # Global Assets, loaded when app starts up, has all Asset and Portfolios 

    close = Col("close")
    open = Col("open")
    high = Col("high")
    low = Col("low")
    atr = Col("atr")
    ema = Col("ema")
    ma = Col("ma")
    tp_perc = Params("tp_perc")
    sl_perc = Params("sl_perc")
    htf_ma = Col("htf_ma")

    # =======================================================================================================|| Global Above

    model_assets=['EURUSD'] # Only keys #, 'GBPUSD'
    model_execution_tf = 'M15'

    strat_param_sets = {
        'AT15': { 
            'execution_tf': model_execution_tf,
            'exit_nb_only_if_pnl_is': 0, 
            'exit_nb_long': range(0, 0+1, 3),
            'exit_nb_short': range(0, 0+1, 3),

            'sl_perc': range(1, 1+1, 1), # 3
            'tp_perc': range(2, 2+1, 1), 
            'param1': range(21, 21+1, 21), #50
            'param2': range(2, 2+1, 1), # 3
            'param3': ['sma'] #, 'ema', 'ema'
        }
    }

    # User imput Indicators
    ind = { 
        'atr': ATR_SL(asset=None, timeframe=model_execution_tf, window='param1'),
        'ema': MA(asset=None, timeframe=model_execution_tf, window='param1', ma_type='ema', price_col='close'),
        'ma': MA(asset=None, timeframe=model_execution_tf, window='param1', ma_type='param3', price_col='close'),
        'htf_ma': MA(asset=None, timeframe='H1', window='param1', ma_type='param3', price_col='close'),
    }

    entry_long = [
        close < open,
        close[1] < open[1],
        close[2] < open[2],
    ]
    entry_short = [
        close > open,
        close[1] > open[1],
        close[2] > open[2],
    ]

    exit_tf_long = [
        close > open,
        close[1] > open[1],
    ]
    exit_tf_short = [
        close < open,
        close[1] < open[1],
    ]

    exit_tp_long_price = [atr * tp_perc]
    exit_tp_short_price = [atr * tp_perc]

    exit_sl_long_price = [atr * sl_perc]
    exit_sl_short_price = [atr * sl_perc]

    entry_long_limit_price = [atr]
    entry_short_limit_price = [atr]

    # --- 1. CORRIGIR strat_num_pos INDIFERENTE, ABRINDO APENAS 1 TRADE
    # --- 2. REIMPLEMENTAR exit_nb_only_if_pnl_is=1/-1
    # 3. CORRIGIR, HTF->LTF Não está funcionado, até porque não está usando a função de calcular indicadores
    # 4. entry_long_limit_price VALE MAIS A PENA COMO POSIÇÃO OU VALOR A SER SOMADO AO OPEN[0] ou CLOSE[1]?
    # 5. IMPLEMENTAR BREAK EVEN POS/NEG 
    # 6. VERIFICAR CADA PARTE DO CÓDIGO E ENTENDER BEM

    AT15 = Strat(
        StratParams(
            name="AT15",
            operation=Backtest(BacktestParams(name='backtest_test')),
            execution_settings=ExecutionSettings(hedge=False, strat_num_pos=[1,1], order_type='market', offset=0.0, 
                                                 day_trade=False, timeTI=None, timeEF=None, timeTF=None, next_index_day_close=False, 
                                                 day_of_week_close_and_stop_trade=[], timeExcludeHours=None, dateExcludeTradingDays=None, dateExcludeMonths=None, 
                                                 fill_method='ffill', fillna=0),
            mma_settings=None, # If mma_rules=None then will use default or PMA or other saved MMA define in Operation. Else it creates a temporary MMA with mma_settings
            params=strat_param_sets['AT15'], # SE signal_params então iterar apenas nos parametros do signal_params para criar sets, else usa apenas sets do indicadores, else sem sets
            indicators=ind,
            signal_rules={
                'entry_long': entry_long,
                'entry_short': entry_short,
                'exit_tf_long': exit_tf_long,
                'exit_tf_short': exit_tf_short,
                'entry_long_limit_price': None,
                'entry_short_limit_price': None,

                'exit_sl_long_price': exit_sl_long_price,
                'exit_sl_short_price': exit_sl_short_price,
                'exit_tp_long_price': exit_tp_long_price,
                'exit_tp_short_price': exit_tp_short_price,

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








