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

    def pre_compute(self, global_assets, timeline, aggr_ret, indicator_pool):
        # Defines parsets from params
        param_sets = self._calculate_param_combinations(self.params)

        # Calculates Indicators 
        if self.indicators: indicator_pool = self._calculate_and_map_indicators(global_assets, timeline, aggr_ret, indicator_pool, param_sets)

        indicator_pool = self._call(self._fn_pre_compute, self._default_pre_compute, global_assets, timeline, aggr_ret, indicator_pool, param_sets)

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

    def _resolve_data_source(self, address, ind_obj, aggr_ret, global_assets, timeline_df, time_col='datetime'):
        # Transforma qualquer string/tupla em um DataFrame [datetime, main]

        # 1. Caso seja um Ativo Físico (ex: "EURUSD")
        if isinstance(address, str) and address in global_assets:
            return global_assets[address].data_get(ind_obj.timeframe)

        # 2. Caso seja um Caminho Exato no sim_data (Tupla)
        if isinstance(address, tuple) and address in aggr_ret:
            return pl.DataFrame({time_col: timeline_df[time_col], "main": aggr_ret[address]})

        # 3. Caso seja uma Query de Agregação (Ex: "@total_long")
        if isinstance(address, str) and address.startswith("@total"):
            side_pref = address.split("_")[-1] if "_" in address else "both"
            target_series = [v for k, v in aggr_ret.items() if k[-1] == side_pref]
            if target_series:
                main_v = pl.DataFrame(target_series).select(pl.mean_horizontal(pl.all())).to_series()
                return pl.DataFrame({time_col: timeline_df[time_col], "main": main_v})
        return None

    def _calculate_and_map_indicators(self, global_assets, timeline, aggr_ret, indicator_pool, param_sets, time_col='datetime'):
        # 1. Timeline Setup (Garantia de ordenação para join_asof)
        timeline_df = (pl.DataFrame({time_col: timeline}) if isinstance(timeline, list) else timeline).sort(time_col)

        for ps_name, ps_dict in param_sets.items():
            for ind_key, ind_obj in self.indicators.items():
                eff_params = self.effective_params_from_global(ind_obj.params, ps_dict)

                # 1. Path resolve
                # Determines final adress list
                raw_addr = ind_obj.asset

                # Case A: Dinamic Expansion (ex: "@each_long", "@each_both")
                if isinstance(raw_addr, str) and raw_addr.startswith("@each"):
                    side_pref = raw_addr.split("_")[-1] if "_" in raw_addr else "both"
                    resolved_addresses = [k for k in aggr_ret.keys() if k[-1] == side_pref]

                # Case B: Manual adresses key
                elif isinstance(raw_addr, list):
                    resolved_addresses = raw_addr

                # Case C: Only 1 adress (Str, Tuple or None)
                else:
                    resolved_addresses = [raw_addr if raw_addr is not None else "@total_both"]

                # 2. Calculation
                for addr in resolved_addresses:
                    indicator_pool.setdefault(ind_key, {}).setdefault(addr, {})

                    if ps_name not in indicator_pool[ind_key][addr]:
                        data_df = self._resolve_data_source(addr, ind_obj, aggr_ret, global_assets, timeline_df)
                        if data_df is not None:
                            res_df = self._calculate_indicator(data_df, ind_obj, eff_params)
                            aligned_series = self._align_IND_to_TIMELINE(timeline_df, res_df, time_col)
                            indicator_pool[ind_key][addr][ps_name] = aligned_series.to_numpy()
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

    def get_ind(self, ind_key, target, i, ps_name=None): # if ps_name=None returns all parsets
        # O(1) search, with parset ensemble support 
        
        try: # Direct O(1) access to global dict tuple (ex: rsi = self.get_ind("rsi_14", "BTCUSD", i))
            ps_dict = self.indicator_pool[ind_key][target]

            # Specific parset request case
            if ps_name is not None:
                return ps_dict[ps_name][i]
            
            # Doesn't have a request but only 1 parset (ex: rsi_score = sum(1 for val in rsi.values() if val > 70))
            if len(ps_dict) == 1:
                return next(iter(ps_dict.values()))[i]
            
            # Case multiple parset and none specifically requested
            return {ps: arr[i] for ps, arr in ps_dict.items()}
        
        except (KeyError, IndexError):
            return None
        
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









