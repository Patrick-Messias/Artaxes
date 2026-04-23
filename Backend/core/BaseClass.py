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
    
    
@dataclass
class BaseManager():

    # ── Helper central ───────────────────────────────────────────────────────

    def _call(self, custom_fn: Optional[Callable], default_fn: Callable, *args, **kwargs):
        # Calls custom_fn if exists, else default
        return custom_fn(*args, **kwargs) if custom_fn else default_fn(*args, **kwargs)
    
    def set_portfolio(self, portfolio_instance):
        self.portfolio = portfolio_instance

    # ── Pre-Simulation ───────────────────────────────────────────────────────

    def pre_compute(self, global_assets, timeline, aggr_ret, indicator_pool, manager_level_key: tuple = ()):
        # manager_level_key: tuple that identifies the level of this manager.
        #     PSM/PMM  → ("portfolio",)
        #     MSM/MMM  → ("op", "model_name")
        #     SSM/SMM  → ("op", "model_name", "strat_name")
        
        # Defines parsets from params
        param_sets = self._calculate_param_combinations(self.params)

        # Calculates Indicators 
        if self.indicators: indicator_pool = self._calculate_and_map_indicators(
            global_assets, timeline, aggr_ret, 
            indicator_pool, param_sets, manager_level_key=manager_level_key)

        indicator_pool = self._call(self._fn_pre_compute, self._default_pre_compute, 
                                    global_assets, timeline, aggr_ret, 
                                    indicator_pool, param_sets, manager_level_key=manager_level_key)

        return indicator_pool, param_sets

    def get_aggr_pnl_by_side(self, df: pl.DataFrame, side: str, alias: str) -> pl.DataFrame:
        # Retorna DataFrame [datetime, pnl] — preserva o datetime para alinhamento posterior.
        if "lot_size" in df.columns:
            if side == "long":
                filtered = df.filter(pl.col("lot_size") > 0)
            elif side == "short":
                filtered = df.filter(pl.col("lot_size") < 0)
            else:
                filtered = df
        else:
            filtered = df

        if filtered.is_empty():
            return pl.DataFrame({"datetime": [], "pnl": []}).with_columns(pl.col("pnl").cast(pl.Float64))

        return (
            filtered
            .group_by("datetime")
            .agg(pl.col("pnl").mean())
            .sort("datetime")
            .rename({"pnl": alias})   # coluna nomeada pelo asset para identificação
        )
    
    # ── Indicators ───────────────────────────────────────────────────────

    def _resolve_data_source(self, address, ind_obj, aggr_ret, global_assets, timeline_df, time_col="datetime"):
        ind_asset = ind_obj.asset

        # Aggregated returns treatment (@each_{} or @total_{})
        if isinstance(ind_asset, str) and (ind_asset.startswith("@each") or ind_asset.startswith("@total")):
            side = ind_asset.split("_")[-1].lower() if "_" in ind_asset else "both"

            # Navegaste aggr nodes
            for _, node_data in aggr_ret.items():
                if isinstance(node_data, dict) and side in node_data:
                    side_data = node_data[side]
                    aggr_price_col = ind_obj.params.get("price_col", "close") if ind_obj.params else "close"
                    
                    # A - @each_{} takes only column reference address (ex: 'AT15')
                    if ind_asset.startswith("@each"):
                        if address in side_data:
                            target_array = side_data[address]
                            return pl.DataFrame({
                                time_col: timeline_df[time_col],
                                aggr_price_col: target_array
                            })
                        
                    # B - @total_{} sums all matrix columns horizontally
                    elif ind_asset.startswith("@total"):
                        total_array = None
                        side_data_size = 0

                        for key, val in side_data.items():
                            if key in ["datetime", "data", "cols", "weights", "time", "ts"]: continue
                            
                            # Sums all data horizontally
                            arr = pl.Series(val)
                            total_array = arr if total_array is None else total_array + arr
                            side_data_size += 1

                        # Applies avg to create portfolio curve
                        if total_array is not None and side_data_size > 0:
                            total_array = total_array / side_data_size

                        return pl.DataFrame({
                            time_col: timeline_df[time_col],
                            aggr_price_col: total_array
                        })

        # Ativos Globais (OHLC bruto)
        if address in global_assets:
            return global_assets[address].load(
                timeframe=ind_obj.timeframe, 
                source="local", 
                date_start=self.portfolio.date_start, 
                date_end=self.portfolio.date_end
            )

        print(f"    < [BaseClass._resolve_data_source] Warning: Address '{address}' not found in global assets or aggregated returns.")
        return None
    
    def _calculate_and_map_indicators(self, global_assets, timeline, aggr_ret, indicator_pool, param_sets, manager_level_key=(), time_col='datetime'):
        # Pool key structure:
        #     (manager_level_key..., ind_key, addr, param_k1, val1, param_k2, val2, ...)
        # Examples:
        #     ("portfolio",         "vol", "@total_both", "window", 21)
        #     ("op", "model",       "vol", "@total_both", "window", 21)   ← different from above
        #     ("op", "model", "s",  "vol", "@total_both", "window", 21)   ← different from both
        #     ("op", "model", "s",  "vol", "EURUSD",      "window", 21)
        
        timeline_df = (pl.DataFrame({time_col: timeline}) if isinstance(timeline, list) else timeline).sort(time_col)
        
        for ps_name, ps_dict in param_sets.items():
            for ind_key, ind_obj in self.indicators.items():
                eff_params = self.effective_params_from_global(ind_obj.params, ps_dict)
                raw_addr = ind_obj.asset

                # --- LÓGICA CORRIGIDA PARA O @EACH COM RECURSIVIDADE ---
                if isinstance(raw_addr, str) and raw_addr.startswith("@each"):
                    side_pref = raw_addr.split("_")[-1].lower() if "_" in raw_addr else "both"
                    resolved_addresses = []

                    for _, valor in aggr_ret.items():
                        if isinstance(valor, dict) and side_pref in valor:
                            side_data = valor[side_pref]
                            if isinstance(side_data, dict):
                                if "cols" in side_data:
                                    resolved_addresses.extend(side_data["cols"])
                                else:
                                    target_keys = [k for k in side_data.keys()
                                                   if k not in {"data", "cols", "weights"}]
                                    resolved_addresses.extend(target_keys)
                    
                    resolved_addresses = list(set(resolved_addresses))
                
                elif isinstance(raw_addr, list):
                    resolved_addresses = raw_addr
                else:
                    resolved_addresses = [raw_addr if raw_addr is not None else "@total_both"]

                # 2. Calculation
                for addr in resolved_addresses:
                    param_items = tuple(
                        item
                        for k in sorted(eff_params.keys())
                        for item in (k, eff_params[k])
                    )
                    # Level prefix garantees PSM and SSM with same ind+addr+params are distinct
                    pool_key = manager_level_key + (ind_key, addr) + param_items
                    if pool_key not in indicator_pool:
                        data_df = self._resolve_data_source(
                            addr, ind_obj, aggr_ret, global_assets, timeline_df)
                        if data_df is not None:
                            res_df  = self._calculate_indicator(data_df, ind_obj, eff_params)
                            aligned = self._align_IND_to_TIMELINE(timeline_df, res_df, time_col)
                            indicator_pool[pool_key] = aligned.to_numpy()
           
        return indicator_pool
    
    def _calculate_indicator(self, data: pl.DataFrame, ind_obj: Indicator, eff_params: dict):
        # Find time column
        time_cols = [c for c in data.columns if c.lower() in ['ts', 'datetime', 'time', 'date']]
        if not time_cols:
            raise ValueError(f"Dataframe for indicator {ind_obj.name} has no valid time column. Columns: {data.columns}")
        time_col = time_cols[0]

        # Case ind uses price_col and == 'close' but only has 'main' uses the latter
        if 'price_col' in eff_params:
            if eff_params['price_col'] not in data.columns and 'main' in data.columns:
                eff_params = eff_params.copy()
                eff_params['price_col'] = 'main'
        #else: print('price_col not in ind')

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


    # Tow below for other uses, _calculate_and_map_indicators already handles Indicator Data to Timeline
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

    def get_ind(self, 
                ind_key:        str, 
                addr:           str = None, 
                idx_start:      int = None,
                idx_end:        int = None,
                level_key:      tuple = None,
                **params
                ):
        
        # >>> Parameters
        # ind_key    : indicator name, e.g. "vol", "ma"
        # addr       : data address, e.g. "EURUSD", "@total_both"  (None = any)
        # level_key  : manager level tuple, e.g. ("portfolio",)    (None = any level)
        # idx_start  : starting index for slicing or single point lookup
        # idx_end    : ending index for slicing
        # **params   : optional param filters, e.g. window=21
        # >>> Returns
        # Single value/array — when exactly 1 match
        # dict {pool_key: value/array} — when multiple matches
        # None — when no match

        matches={}

        for k, v in self.portfolio.indicator_pool.items():
            # Searches ind_key position in the tuple
            try:
                ind_pos = k.index(ind_key)
            except ValueError:
                continue

            # addr is always right after ind_key
            k_addr = k[ind_pos + 1] if ind_pos + 1 < len(k) else None

            # level_key is everything before ind_key
            k_level = k[:ind_pos]

            # params are everything after addr, in pairs
            k_params_flat = k[ind_pos + 2:]
            k_params = {str(k_params_flat[i]): str(k_params_flat[i + 1]) 
                        for i in range(0, len(k_params_flat) - 1, 2)}

            # Apply filters
            if addr      and k_addr  != addr:       continue
            if level_key and k_level != level_key:  continue
            if params:
                match_params = True
                for pk, pv in params.items():
                    if str(k_params.get(pk)) != str(pv):
                        match_params = False
                        break 
                if not match_params:
                    continue

            matches[k] = v

        if not matches:
            print(f"    < [BaseManager.get_ind] Warning: No matches found for ind_key='{ind_key}', addr='{addr}', level_key='{level_key}', params={params}.")
            return None
        
        def _slice_data(data_array):
            vals = data_array[:,1] if data_array.ndim > 1 else data_array

            if idx_start is not None and idx_end is not None:
                return vals[idx_start:idx_end]
            elif idx_start is not None:
                return vals[idx_start]
            elif idx_end is not None:
                return vals[:idx_end]
            return vals
        
        # Applies slicing for all matches
        sliced_matches = {k: _slice_data(v) for k, v in matches.items()}
        
        if len(sliced_matches) == 1:
            return next(iter(sliced_matches.values()))
        return sliced_matches
    
    '''
            for k, v in self.portfolio.indicator_pool.items():
            # Locate ind_key in the tuple
            ind_pos = None
            for pos, elem in enumerate(k):
                if elem == ind_key:
                    ind_pos = pos
                    break
            if ind_pos is None:
                continue

            # addr is always right after ind_key
            k_addr = k[ind_pos + 1] if ind_pos + 1 < len(k) else None

            # level_key is everything before ind_key
            k_level = k[:ind_pos]

            # params are everything after addr, in pairs
            k_params_flat = k[ind_pos + 2:]
            k_params = {k_params_flat[i]: k_params_flat[i + 1]
                        for i in range(0, len(k_params_flat) - 1, 2)}

            # Apply filters
            if addr      and k_addr  != addr:       continue
            if level_key and k_level != level_key:  continue
            if params:
                if not all(k_params.get(pk) == pv for pk, pv in params.items()):
                    continue

            matches[k] = v

        if not matches:
            return None
        
        def _slice_data(data_array):
            vals = data_array[:,1] if data_array.ndim > 1 else data_array

            if idx_start is not None and idx_end is not None:
                return vals[idx_start:idx_end]
            elif idx_start is not None:
                return vals[idx_start]
            elif idx_end is not None:
                return vals[:idx_end]
            return vals
        
        # Applies slicing for all matches
        sliced_matches = {k: _slice_data(v) for k, v in matches.items()}

        if len(matches) == 1:
            return next(iter(sliced_matches.values()))
        return sliced_matches
    '''

    def get_data(self, key=None, lookback=1, data_type="aggr", side="BOTH"):
        # Aux method for managers to search data
        if key is None: key = (self.portfolio,)

        if lookback is None or lookback == 0:
            start_idx = 0
        else:
            start_idx = max(0, self.portfolio.current_idx - lookback+1)

        return self.portfolio._populate_sim_data(
            key=key,
            i=self.portfolio.current_idx,
            start_idx=start_idx,
            data_type=data_type,
            side=side
        )

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









