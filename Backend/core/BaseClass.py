from itertools import product
from dataclasses import dataclass
import polars as pl, json
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
    
    # ── Pre-Simulation ───────────────────────────────────────────────────────

    def pre_compute(self, timeline, sim_data, aggr_ret, indicator_pool) -> None:
        self._call(self._fn_pre_compute, self._default_pre_compute, timeline, sim_data, aggr_ret, indicator_pool)

    # ── Rebalance Func ───────────────────────────────────────────────────────

    def rank(self, context: dict) -> Dict[str, float]:
        # Ranks each model by metric defined in model_hierarchy. Returns dict[model_name: score]
        return self._call(self._fn_rank, self._default_rank, context)

    def filter(self, context: dict) -> List[str]:
        # Removes models that don't pass the filter function
        # Returns list of model_names that are active
        return self._call(self._fn_filter, self._default_filter, context)

    def rebalance(self, context: dict) -> List[str]:
        # Orchestrates rank -> filter -> selection
        # Returns ordered list of active models
        return self._call(self._fn_rebalance, self._default_rebalance, context)

    # ── Indicators ───────────────────────────────────────────────────────

    def _calculate_and_map_indicators(self, aggr_ret, indicator_pool, param_sets):
        # USE EXAMPLE
        # def check_entry(self, i, ps_name, indicator_pool):
        #     # Acessa pelo nome que você deu na definição, sem se preocupar com o hash
        #     u_key = self._pre_cache[ps_name]["trend_ma"]
        #     current_ma = indicator_pool[u_key]["main"][i]
        #     if current_pnl > current_ma: return True
        #
        # u_key = self.psm._pre_cache[ps_name]["rsi"]
        # rsi_val = indicator_pool[u_key]["main"][i]

        # Creates 1 dataframe with all models results
        if aggr_ret:
            all_models_df = pl.DataFrame(aggr_ret)
            portfolio_series = all_models_df.select(pl.mean_horizontal(pl.all())).to_series()

        for _, ps_dict in param_sets.items():
            for ind_key, ind_obj in self.sm_indicators.items():
                eff_params = self.effective_params_from_global(ind_obj.params, ps_dict)
                ind_p_hash = self.param_suffix(eff_params)

                if ind_obj.asset is None: # Calculates for each Asset in sm_assets
                    if self.sm_assets is None: continue
                    
                    for a_name, a_obj in self.sm_assets:
                        unique_key = f"{a_name}_{ind_obj.timeframe}_{ind_key}_{ind_p_hash}"
                        if unique_key not in indicator_pool: 
                            asset_df = a_obj.data_get(ind_obj.timeframe)
                            if asset_df is not None: 
                                indicator_pool[unique_key] = self._calculate_indicator(asset_df, ind_obj, eff_params)
                else:
                    if aggr_ret is None: continue

                    if ind_obj.asset == "model": # Calculates for each model in aggr_models_ret
                        for m_name, m_obj_series in aggr_ret.items():
                            unique_key = f"model_{m_name}_{ind_obj.timeframe}_{ind_key}_{ind_p_hash}"
                            if unique_key not in indicator_pool: 
                                indicator_pool[unique_key] = self._calculate_indicator(m_obj_series, ind_obj, eff_params)

                    elif ind_obj.asset == "models": # Calculates with sum of all models in aggr_models_ret
                        unique_key = f"portfolio_total_{ind_obj.timeframe}_{ind_key}_{ind_p_hash}"
                        if unique_key not in indicator_pool: 
                            indicator_pool[unique_key] = self._calculate_indicator(portfolio_series, ind_obj, eff_params)
                
        # Maps Indicators to _pre_cache with all parsets
        for ps_name, ps_dict in param_sets.items():
            self._pre_cache[ps_name] = {}
            for ind_key, ind_obj in self.sm_indicators.items():
                eff_params = self.effective_params_from_global(ind_obj.params, ps_dict)
                ind_p_hash = self.param_suffix(eff_params)

                u_key = f"model_{self.name}_{ind_obj.timeframe}_{ind_key}_{ind_p_hash}"
                self._pre_cache[ps_name][ind_key] = u_key

        return indicator_pool
    
    def _calculate_indicator(self, data, ind_obj, eff_params):
        # data: Can be pl.DataFrame (OHLC) or pl.Series (PnL Aggregated / Tick)
        # eff_params: Dict with params that only this indicator uses

        # Executes Indicator, ind_obj must be able to accept Series + params
        ind_results = ind_obj.calculate(data, eff_params)

        if isinstance(ind_results, pl.Series):
            return {"main": ind_results}

        return ind_results

    # ── Global Func ───────────────────────────────────────────────────────

    def get_schedule(self, timeline: list) -> set:
        freq = self.reb_frequency 

        if not freq or freq == "never": 
            return pl.DataFrame({"ts": None}) # Updates every datetime

        df = pl.DataFrame({"ts": timeline})

        if freq == "tick":
            return df # Will always run

        if freq == "daily":
            condition = pl.col("ts").dt.date() != pl.col("ts").dt.date().shift(1)
        if freq == "weekly":
            condition = pl.col("ts").dt.week() != pl.col("ts").dt.week().shift(1)
        elif freq == "monthly":
            condition = pl.col("ts").dt.month() != pl.col("ts").dt.month().shift(1)
        elif freq == "yearly":
            condition = pl.col("ts").dt.year() != pl.col("ts").dt.year().shift(1)
        else:
            return set()

        # Fist candle is always a point of rebalance (start)
        return set(df.filter(condition | pl.col("ts").is_first())["ts"].to_list())

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









