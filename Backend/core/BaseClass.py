from itertools import product
from dataclasses import dataclass
import polars as pl, json, Indicator
from typing import Dict, Optional, Callable, List

@dataclass
class BaseClass():
    # Calculates param_set(s) for optimization based on a given input dictionary.
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
    
    @staticmethod
    def separate_long_short_returns(df: pl.DataFrame) -> pl.DataFrame:
        # df needs at least: 'datetime', 'return', 'lot'
        
        # If dict with DataFrames
        if isinstance(df, dict):
            return {k: BaseClass.separate_long_short_returns(v) for k, v in df.items()}
        
        if isinstance(df, pl.DataFrame):
            if "lot" not in df.columns: 
                return df
            
            return df.with_columns([
                pl.when(pl.col("lot") > 0).then(pl.col("return")).otherwise(0.0).alias("long_return"),
                pl.when(pl.col("lot") < 0).then(pl.col("return")).otherwise(0.0).alias("short_return"),
                pl.when(pl.col("lot") > 0).then(1).when(pl.col("lot") < 0).then(-1).otherwise(0).alias("position_direction")
            ])
        return df

@dataclass
class BaseManager():

    # ── Helper central ───────────────────────────────────────────────────────

    def _call(self, custom_fn: Optional[Callable], default_fn: Callable, *args, **kwargs):
        # Calls custom_fn if exists, else default
        return custom_fn(*args, **kwargs) if custom_fn else default_fn(*args, **kwargs)
    
    # ── Pre-Simulation ───────────────────────────────────────────────────────

    def pre_compute(self, global_assets, timeline, sim_data, aggr_ret, indicator_pool):
        # Defines parsets from params
        param_sets = self._calculate_param_combinations(self.params)

        # Calculates Indicators 
        if self.indicators: indicator_pool = self._calculate_and_map_indicators(global_assets, timeline, aggr_ret, indicator_pool, param_sets)

        indicator_pool, sim_data = self._call(self._fn_pre_compute, self._default_pre_compute, global_assets, timeline, sim_data, aggr_ret, indicator_pool, param_sets)

        return indicator_pool, sim_data, param_sets

    # ── Indicators ───────────────────────────────────────────────────────

    def get_ind(self, ind_key: str, target: str, i: int, indicator_pool: dict, ps_name: str, col: str = "main"):
        """
        Searches in O(1) indicator value aligned to the current iteration
        
        :param ind_key: Indicator name (ex: "rsi", "trend_ma")
        :param target: Asset/Model name (ex: "BTC"), model (ex: "Model_A") or "total"
        :param i: Timeline iteration current index 
        :param indicator_pool: Global dictionary that holds aligned dfs
        :param ps_name: Parset name (ex: "default")
        :param col: Indicator column to select (default is "main")
        """
        try:
            # 1. Recovers unique cache hash
            u_key = self._pre_cache[ps_name][ind_key][target]
            
            # 2. Polars DataFrame Access line 'i' of the selected column
            return indicator_pool[u_key].get_column(col).item(i)
            
        except (KeyError, ValueError): # Case indicator doesn't exist for this combination
            return None
        except IndexError: # Case iteration higher then array size
            return None

    def _calculate_and_map_indicators(self, global_assets, timeline, aggr_ret, indicator_pool, param_sets):
        # 1. Timeline Setup
        if isinstance(timeline, list):
            timeline_df = pl.DataFrame({"datetime": timeline})
        else:
            timeline_df = timeline

        time_col = next((c for c in timeline_df.columns if c.lower() in ['ts', 'datetime', 'time', 'date']), timeline_df.columns[0])
        
        # GARANTIA OBRIGATÓRIA: A timeline precisa estar ordenada para o join_asof não quebrar
        timeline_df = timeline_df.sort(time_col)

        # 2. Portfolio/Aggr Setup
        portfolio_series = None
        if aggr_ret:
            # Converte chaves (tuplas) para strings para o Polars aceitar como colunas
            str_aggr_ret = {
                "-".join(map(str, k)) if isinstance(k, (tuple, list)) else str(k): v 
                for k, v in aggr_ret.items()
            }
            
            all_models_df = pl.DataFrame(str_aggr_ret)
            portfolio_series = pl.DataFrame({
                time_col: timeline_df.get_column(time_col),
                "main": all_models_df.select(pl.mean_horizontal(pl.all())).to_series()
            })

        # 3. Loop de Parsets e Indicadores
        for ps_name, ps_dict in param_sets.items():
            self._pre_cache[ps_name] = {}

            for ind_key, ind_obj in self.indicators.items():
                eff_params = self.effective_params_from_global(ind_obj.params, ps_dict)
                ind_p_hash = self.param_suffix(eff_params)
                self._pre_cache[ps_name][ind_key] = {}

                # --- LÓGICA POR ATIVO ---
                if ind_obj.asset is None or (not isinstance(ind_obj.asset, str) or ind_obj.asset not in ["each_aggr", "all_aggr"]):
                    if self.assets is None and ind_obj.asset is None: continue
                    target_assets = ind_obj.asset if isinstance(ind_obj.asset, list) else ([ind_obj.asset] if ind_obj.asset else self.assets)

                    for a_name in target_assets:
                        a_obj = global_assets.get(a_name)
                        if not a_obj: continue

                        unique_key = f"{a_name}_{ind_obj.timeframe}_{ind_key}_{ind_p_hash}"
                        self._pre_cache[ps_name][ind_key][a_name] = unique_key
                        
                        if unique_key not in indicator_pool: 
                            asset_df = a_obj.data_get(ind_obj.timeframe)
                            if asset_df is not None: 
                                ind_result = self._calculate_indicator(asset_df, ind_obj, eff_params)
                                indicator_pool[unique_key] = self._align_IND_to_TIMELINE(timeline_df, ind_result, time_col)

                # --- LÓGICA AGREGADA ---
                elif aggr_ret:
                    if ind_obj.asset == "each_aggr": 
                        for m_key, m_obj_series in aggr_ret.items():
                            m_str = "-".join(map(str, m_key)) if isinstance(m_key, (tuple, list)) else str(m_key)
                            unique_key = f"each_aggr_{m_str}_{ind_obj.timeframe}_{ind_key}_{ind_p_hash}"

                            self._pre_cache[ps_name][ind_key][m_key] = unique_key

                            if unique_key not in indicator_pool: 
                                m_df = pl.DataFrame({
                                    time_col: timeline_df.get_column(time_col),
                                    "main": m_obj_series
                                })
                                ind_result = self._calculate_indicator(m_df, ind_obj, eff_params)
                                indicator_pool[unique_key] = self._align_IND_to_TIMELINE(timeline_df, ind_result, time_col)

                    elif ind_obj.asset == "all_aggr" and portfolio_series is not None: 
                        unique_key = f"all_aggr_{ind_obj.timeframe}_{ind_key}_{ind_p_hash}"
                        self._pre_cache[ps_name][ind_key]["total"] = unique_key

                        if unique_key not in indicator_pool: 
                            ind_result = self._calculate_indicator(portfolio_series, ind_obj, eff_params)
                            indicator_pool[unique_key] = self._align_IND_to_TIMELINE(timeline_df, ind_result, time_col)
                
        return indicator_pool
    
    def _calculate_indicator(self, data: pl.DataFrame, ind_obj: Indicator, eff_params: dict):
        # Find time column
        time_col = next(c for c in data.columns if c.lower() in ['ts', 'datetime', 'time', 'date']) 

        # Case ind uses price_col and == 'close' but only has 'main' uses the latter
        if 'price_col' in eff_params:
            if eff_params['price_col'] not in data.columns and 'main' in data.columns:
                eff_params = eff_params.copy()
                eff_params['price_col'] = 'main'
                print(f"ind {eff_params['price_col']}")
        else: print('price_col not in ind')

        # Gets indicator expression
        exprs = ind_obj.get_expression(eff_params)

        # Executes expression in native and optimized form
        if isinstance(exprs, list): # For mult lines ind returns DataFrame with time + columns
            return data.select([pl.col(time_col)] + exprs)
        else: # For singles indicator
            return data.select([
                pl.col(time_col),
                exprs.alias("main")
            ])

    def _align_IND_to_TIMELINE(self, timeline_df: pl.DataFrame, indicator_df: pl.DataFrame, time_col="datetime", fills_nulls="forward") -> pl.DataFrame:
        
        if time_col not in indicator_df.columns:
            raise f"{time_col} not in Indicator DataFrame columns"

        # Applies shift to avoid lookahead bias
        indicator_shifted = indicator_df.with_columns(
            pl.all().exclude(time_col).shift(1)
        )

        # Join Asof (Backward) each timeline date gets last indicator value lower or equal to him
        aligned_df = timeline_df.join_asof(
            indicator_shifted,
            on=time_col,
            strategy="backward"
        )

        # Fills nulls with zero
        if fills_nulls == 0: return aligned_df.fill_null(0)
        elif fills_nulls == "forward": return aligned_df.fill_null(strategy="forward")

    # Tow below for other use, _calculate_and_map_indicators already handles Indicator Data to Timeline
    def align_HTF_to_LTF(timeline_df, higher_tf_df): 
        # timeline_df: DataFrame com a coluna 'ts' (ex: 1 min ou ticks)
        # higher_tf_df: DataFrame com o indicador calculado (ex: 1 hora)

        # 1. Aplicamos o shift no indicador de TF maior
        # O valor calculado na vela que FECHA às 02:00 só é válido para o futuro
        higher_tf_prepared = higher_tf_df.with_columns(
            pl.all().exclude("ts").shift(1)
        )

        # 2. Join Asof: Une os dados onde timeline['ts'] >= higher_tf['ts']
        # Ele pega o último valor disponível (backward fill automático)
        aligned_df = timeline_df.join_asof(
            higher_tf_prepared,
            on="ts",
            strategy="backward"
        )
        
        return aligned_df.fill_null(strategy="forward") # Opcional: preenche o início do histórico
    
    def align_LTF_to_HTF(df_ltf, htf_period="1h", method="last"): 
        # Consolida dados do menor timeframe para o maior.
        # method: "last", "mean", "sum", "max", "min"
        
        aggs = {
            "last": pl.col("valor").last(),
            "mean": pl.col("valor").mean(),
            "sum":  pl.col("valor").sum(),
            "max":  pl.col("valor").max()
        }
        
        return df_ltf.group_by_dynamic(
            "ts", 
            every=htf_period, 
            closed="right" # Garante que a vela das 02:00 inclua o dado de 02:00
        ).agg(aggs.get(method, pl.col("valor").last()))

    # ── Global Func ───────────────────────────────────────────────────────

    def get_schedule(self, timeline: list) -> set:
        freq = self.reb_frequency 

        if not freq or freq == "never": 
            return set() # Never runs

        df = pl.DataFrame({"ts": timeline})

        if freq == "tick":
            return set(timeline) # Will always run
        
        if freq == "daily":
            condition = pl.col("ts").dt.date() != pl.col("ts").dt.date().shift(1)
        elif freq == "weekly":
            condition = pl.col("ts").dt.week() != pl.col("ts").dt.week().shift(1)
        elif freq == "monthly":
            condition = pl.col("ts").dt.month() != pl.col("ts").dt.month().shift(1)
        elif freq == "yearly":
            condition = pl.col("ts").dt.year() != pl.col("ts").dt.year().shift(1)
        else:
            return set()
        
        # Uses fill_null(False) to garantee that shift(1) null doesn't break the filter
        filtered_dates = df.filter(condition.fill_null(False))["ts"].to_list()

        # Converts to Set
        schedule_set = set(filtered_dates)

        # Forces first candle to always be be a rebalance point
        if timeline:
            schedule_set.add(timeline[0])

        return schedule_set

    #||=========================================================================================||








# def main():
#     # 1. Instancia a base
#     tester = BaseClass()

#     # 2. Define o dicionário de entrada com parâmetros FIXED e VARYING
#     # Imagine um cenário de backtesting de estratégia
#     config = {
#         "timeframe": "H4",             # Fixed
#         "ema_period": [21, 42 +1, 21],     # Varying (list)
#         "rsi_threshold": range(2, 3 +1),# Varying (range - aqui gera apenas 1, mas é iterável)
#         "multiplier": (1.5, 2.0),      # Varying (tuple)
#     }

#     print(f"--- Iniciando geração de combinações ---")
#     print(f"Input: {len(config)} chaves detectadas.\n")

#     # 3. Calcula as combinações
#     param_sets = tester._calculate_param_combinations(config, prefix="STRAT")

#     # 4. Printa os resultados de forma organizada
#     print(f"Total de combinações geradas: {len(param_sets)}\n")
    
#     for i, (name, params) in enumerate(param_sets.items(), 1):
#         print(f"[{i}] ID: {name}")
#         print(f"    Params: {params}")
#         print("-" * 30)

#     # Exemplo de como isso seria usado para salvar um JSON ou alimentar o DuckDB
#     # print("\nEstrutura final (JSON Style):")
#     # print(json.dumps(param_sets, indent=4))

# if __name__ == "__main__":
#     main()









