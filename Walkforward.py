import polars as pl, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import List, Literal, Dict, Any, Union

# Defines types to facilitate reuse
SortOrder = Literal['asc', 'des']

class Walkforward:
    def __init__(
        self,
        matrix: pl.DataFrame=None,
        wfm_configs: List[List[int]]=[[12, 4, 4]], # [[IS, OS, STEP], ...]
        matrix_resolution: Literal['daily', 'weekly', 'monthly'] = 'weekly',

        # Parset Selection
        is_metric: Literal['pnl', 'sharpe', 'pnl_dd'] = 'pnl',
        is_top_n: int=1,
        is_logic: Literal['highest', 'mode'] = 'highest',
        is_order: SortOrder = 'des',

        # Final WF Seletion from WFM
        wf_selection_metric: Literal['pnl', 'pnl_dd', 'wfe', 'wfe_sharpe', 'avg_sharpe', 'wfe_mean', 'wfe_sharpe'] = 'pnl',
        wf_selection_analysis_radius_n: int=2,
        wf_selection_logic: Literal['highest', 'highest_stable', 'stable'] = 'highest',
        wf_returns_mode: Literal['selected', 'all'] = 'selected'
    ):
        # A matrix já vem pronta do seu _process_wfm_to_polars no Operation
        self.matrix = matrix 
        self.wfm_configs = wfm_configs
        self.matrix_resolution = matrix_resolution

        self.is_metric = is_metric
        self.is_top_n = is_top_n
        self.is_order = is_order
        self.is_logic = is_logic

        self.wf_selection_metric = wf_selection_metric
        self.wf_selection_analysis_radius_n = wf_selection_analysis_radius_n
        self.wf_selection_logic = wf_selection_logic
        self.wf_returns_mode = wf_returns_mode

        self.all_wf_results = {}

    def _find_best_parameter(self, is_data: pl.DataFrame) -> str:
        if is_data.is_empty(): return None

        # Unpacks WF Parset selection config
        is_descending = False if self.is_logic == 'asc' else True

        # Calculates metrics for all columns (ps_ids)
        ps_cols = [c for c in is_data.columns if c != "ts"]
        
        if self.is_metric == "pnl":
            stats = is_data.select([pl.col(c).sum().alias(c) for c in ps_cols])
        elif self.is_metric == "pnl_dd": #PnL acumulado / Range da curva (Proxy de Drawdown)
            stats = is_data.select([
                (pl.col(c).sum() / (pl.col(c).cumsum().max() - pl.col(c).cumsum().min() + 1e-6)).alias(c)
                for c in ps_cols
            ])
        elif self.is_metric == "sharpe":
            stats = is_data.select([
                (pl.col(c).mean() / (pl.col(c).std() + 1e-9)).alias(c) 
                for c in ps_cols
            ])
        else: # Default is PnL only
            stats = is_data.select([pl.col(c).sum().alias(c) for c in ps_cols])

        # Ranks and takes top N
        ranked_ps = stats.unpivot(variable_name="ps_id", value_name="val")
        ranked_ps = ranked_ps.sort(["val", "ps_id"], descending=[is_descending, False]).head(self.is_top_n)
        top_ps_list = ranked_ps["ps_id"].to_list()

        if not top_ps_list: return None

        # Selection Logic
        if self.is_logic == 'highest' or self.is_logic == 'h' or self.is_logic == 'rank': 
            return top_ps_list[0]
        if self.is_logic == 'moda' or self.is_logic == 'mode': # Decomposes strings and to finds out parset
            param_counts = {}
            for ps_id in top_ps_list: # Removes prefix param_set-, if split with "-"
                params = tuple(ps_id.replace("ps_param_set-", "").replace("ps_", "").split("-"))
                param_counts[params] = param_counts.get(params, 0) + 1

            # Mode is the tuple that showed up the most
            best_params_tuple = max(param_counts, key=param_counts.get)

            # Reconstructs original ps_id
            for ps_id in top_ps_list:
                if tuple(ps_id.replace("ps_param_set-", "").replace("ps_", "").split("-")) == best_params_tuple:
                    return ps_id
        
        return top_ps_list[0]

    def run_wf(self, is_periods: int, os_periods: int, step_periods: int = None):
        if step_periods is None: step_periods = os_periods

        df = self._temp_df
        total_rows = len(df)
        runs = []

        # Navegates by index (Lines) instead of Timedeltas
        start_idx = 0
        current_is_end_idx = is_periods

        while current_is_end_idx < total_rows:
            current_oos_end_idx = min(current_is_end_idx + os_periods, total_rows)

            # Garantees that OOS matrix limits aren't breached  
            if current_oos_end_idx > total_rows:
                current_oos_end_idx = total_rows

            # Slices the Matrix by positions
            is_data = df.slice(start_idx, is_periods)
            os_data = df.slice(current_is_end_idx, (current_oos_end_idx - current_is_end_idx))

            # Selects best param IS
            best_ps_id = self._find_best_parameter(is_data)

            if best_ps_id is not None and not os_data.is_empty(): # Extracts values, making shure they are scalars with .item()
                os_returns = os_data[best_ps_id]

                is_pnl_total = float(is_data[best_ps_id].sum() or 0.0)
                os_pnl_total = float(os_returns.sum() or 0.0)

                # WFE (Avg OS / Avg IS)
                wfe = (os_pnl_total / os_periods) / (is_pnl_total / is_periods) if is_pnl_total != 0 else 0

                # Adjust Sharpe's annualized factor according to resolution
                sharpe_ann_factor = 252 if self.matrix_resolution == 'daily' else (52 if self.matrix_resolution == 'weekly' else 12)
                m = os_returns.mean()
                s = os_returns.std()
                mean_val = float(m) if m is not None else 0.0
                std_val = float(s) if s is not None else 0.0

                os_sharpe = (mean_val / (std_val + 1e-9)) * np.sqrt(sharpe_ann_factor) 

                runs.append({
                    "os_curve": os_data.select(["ts", best_ps_id]).rename({best_ps_id: "pnl"}),
                    "wfe": wfe,
                    "metrics": {
                        "is_pnl": is_pnl_total,
                        "os_pnl": os_pnl_total,
                        "os_sharpe": os_sharpe,
                        "best_ps": best_ps_id
                    }
                })
            
            # Advances window by step_periods
            start_idx += step_periods
            current_is_end_idx += step_periods

            # Ends if next IS doen't fit in matrix
            if start_idx + is_periods > total_rows: break

        return runs


    def analyze(self):
        # Clears previous results to avoid leakage
        self.all_wf_results = {} 

        # Keeps original matrix intact
        working_df = self.matrix.clone()

        if self.matrix_resolution == 'weekly':
            working_df = (
                working_df.upsample(time_column="ts", every="1d").fill_null(0)
                .group_by_dynamic("ts", every="1w", closed="left")
                .agg(pl.all().exclude("ts").sum())
            )
        elif self.matrix_resolution == 'monthly':
            working_df = (
                working_df.group_by_dynamic("ts", every="1mo", closed="left")
                .agg(pl.all().exclude("ts").sum())
            )

        # Stores matrix temp for run_wf
        self._temp_df = working_df.sort("ts")

        for config in self.wfm_configs:
            is_l, os_l, stp = config
            wf_key = f"IS{is_l}_OS{os_l}"

            runs = self.run_wf(is_l, os_l, stp)
            if not runs:
                continue
        
            full_oos_curve = pl.concat([r["os_curve"] for r in runs]).sort("ts")

            # Max Drawdown da curva OOS
            cum_pnl = full_oos_curve["pnl"].cum_sum()
            wfes = [r["wfe"] for r in runs]

            total_pnl_value = float(full_oos_curve["pnl"].sum() or 0.0)
            max_dd_val = float((cum_pnl.cum_max() - cum_pnl).max() or 0.0)
            avg_sharpe = float(np.mean([r["metrics"]["os_sharpe"] for r in runs]) or 0.0)
            wfe_mean = float(np.mean(wfes) or 0.0)
            wfe_sharpe = float(np.mean(wfes) / (np.std(wfes) + 1e-9) or 0.0)

            self.all_wf_results[wf_key] = {
                "curve": full_oos_curve,
                "total_pnl": total_pnl_value,
                "max_dd": max_dd_val,
                "avg_sharpe": avg_sharpe,
                "wfe_mean": wfe_mean,
                "wfe_sharpe": wfe_sharpe,
                "runs": runs,
                "config": config
            }
        
        return self._finalize()

    def _finalize(self):
        if not self.all_wf_results:
            print("      > [Warning] No Walkforward results to finalize")
            return None
        
        metric_key = 'total_pnl' if self.wf_selection_metric == 'pnl' else self.wf_selection_metric

        # If highest and selected then cuts here
        if self.wf_selection_logic == 'highest' and self.wf_returns_mode == 'selected':
            best_key = max(
                self.all_wf_results,
                key=lambda k: self.all_wf_results[k].get(metric_key, -np.inf)
            )
            return self.all_wf_results[best_key]
            
        # Crates a coordinates (IS, OS, STEP) search dict
        grid = {}
        is_vals = sorted(list(set(res['config'][0] for res in self.all_wf_results.values())))
        os_vals = sorted(list(set(res['config'][1] for res in self.all_wf_results.values())))
        st_vals = sorted(list(set(res['config'][2] for res in self.all_wf_results.values())))

        for key, res in self.all_wf_results.items():
            conf = res['config'] # [IS, OS, ST]
            grid[tuple(conf)] = res.get(metric_key, -np.inf)

        # Calculates stability score
        final_scores = {}
        r = self.wf_selection_analysis_radius_n

        for key, res in self.all_wf_results.items():
            conf = tuple(res['config'])
            current_val = res.get(metric_key, -np.inf)

            # Keeps here in case, in case highest score is the brute value
            if self.wf_selection_logic == 'highest':
                final_scores[key] = current_val
                continue

            # For 'stable' or 'highest_stable' calculates neighborhood average
            neighborhood_values = []
            ix = is_vals.index(conf[0])
            io = os_vals.index(conf[1])
            is_ = st_vals.index(conf[2])

            # Runs through neighborhood cube defined by n radious
            for dx in range(-r, r+1):
                for do in range(-r, r+1):
                    for ds in range(-r, r+1): 
                        try: # Tries to find neighbor in adjusted coordinates
                            if (ix+dx < 0) or (io+do < 0) or (is_+ds < 0): continue
                            neighbor_conf = (is_vals[ix+dx], os_vals[io+do], st_vals[is_+ds])
                            if neighbor_conf in grid:
                                neighborhood_values.append(grid[neighbor_conf])
                        except IndexError:
                            continue
            avg_neighbor = np.mean(neighborhood_values) if neighborhood_values else current_val

            if self.wf_selection_logic == 'stable':
                final_scores[key] = avg_neighbor
            elif self.wf_selection_logic == 'highest_stable':
                penalty = (avg_neighbor / (abs(avg_neighbor) + 1e-9))
                final_scores[key] = current_val * penalty if current_val > 0 else current_val

        if self.wf_returns_mode == 'selected': # Returns walkforward with highest metric from all wfm
            best_key = max(final_scores, key=final_scores.get)
            return self.all_wf_results[best_key]
        if self.wf_returns_mode == 'all':
            for k in self.all_wf_results:
                self.all_wf_results[key]['stability_score'] = final_scores.get(key, 0)
        return self.all_wf_results

    # Plotting
    def plot_heatmap(self):
        # 1. Aplicar o estilo escuro do Matplotlib
        plt.style.use('dark_background')

        data = []
        for key, res in self.all_wf_results.items():
            is_l, os_l = res['config'][0], res['config'][1]
            data.append({"IS": is_l, "OS": os_l, "PnL": res['total_pnl']})

        df_plot = pd.DataFrame(data).pivot(index="IS", columns="OS", values="PnL")
        df_plot = df_plot.sort_index(ascending=False)

        # 2. Configurar o Seaborn para contexto escuro
        # 'darkgrid' ou 'white' (com dark_background o white fica transparente/escuro)
        sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#121212", "figure.facecolor": "#121212"})

        plt.figure(figsize=(12, 8))
        
        # 3. Gerar o heatmap
        # cmap="RdYlGn" continua bom, mas você pode usar "viridis" ou "magma" para um look mais 'cyberpunk'
        ax = sns.heatmap(
            df_plot, 
            annot=True, 
            fmt=".2f", 
            cmap="RdYlGn", 
            center=0,
            annot_kws={"size": 10, "weight": "bold"}, # Melhora leitura no escuro
            linewidths=.5, 
            linecolor='#333333' # Cor da divisão das células
        )

        plt.title("WFM Matrix - Robustness Analysis (Dark Theme)", color='white', fontsize=15)
        plt.xlabel("Out-of-Sample Size (OS)", color='white')
        plt.ylabel("In-Sample Size (IS)", color='white')
        
        # Ajustar as cores das labels dos eixos
        plt.xticks(color='white')
        plt.yticks(color='white')

        plt.show()
        
        # Opcional: Resetar para o tema padrão após o plot para não afetar outros gráficos
        plt.style.use('default')

    def plot_advanced_heatmap(self, metric: Literal['total_pnl', 'max_dd', 'avg_sharpe', 'wfe_sharpe'] = 'total_pnl'):
        # 1. Aplicar o estilo escuro
        plt.style.use('dark_background')
        sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#121212", "figure.facecolor": "#121212"})
        
        data = [{"IS": r['config'][0], "OS": r['config'][1], "Val": r[metric]} for r in self.all_wf_results.values()]
        df_plot = pd.DataFrame(data).pivot(index="IS", columns="OS", values="Val").sort_index(ascending=False)

        plt.figure(figsize=(12, 8))
        
        # Inverter cores se a métrica for Drawdown (Menor queda é melhor -> Verde)
        cmap = "RdYlGn_r" if metric == 'max_dd' else "RdYlGn"
        
        # 2. Gerar o heatmap com os mesmos parâmetros visuais do plot_heatmap
        ax = sns.heatmap(
            df_plot, 
            annot=True, 
            fmt=".3f", 
            cmap=cmap, 
            center=0 if metric != 'max_dd' else None,
            annot_kws={"size": 10, "weight": "bold"},
            linewidths=.5, 
            linecolor='#333333'
        )

        # 3. Adicionar labels e estilização dos eixos (X e Y)
        plt.title(f"WFM Matrix - {metric.upper()} (Dark Theme)", color='white', fontsize=15)
        plt.xlabel("Out-of-Sample Size (OS)", color='white')
        plt.ylabel("In-Sample Size (IS)", color='white')
        
        # Garantir que os números nos eixos fiquem brancos
        plt.xticks(color='white')
        plt.yticks(color='white')

        plt.show()
        
        # Resetar para o padrão
        plt.style.use('default')

    def plot_wfe_analysis(self): # Plots WFE Efficiency timeline
        plt.figure(figsize=(12, 6))
        for wf_key, res in self.all_wf_results.items():
            wfes = [r['wfe'] for r in res['runs']]
            dates = [r['os_curve']['datetime'].min() for r in res['runs']]
            plt.plot(dates, wfes, label=f"{wf_key} (Avg: {res['wfe']:.2f})")
        
        plt.axhline(y=1.0, color='r', linestyle='--', label='Perfect Efficiency (1.0)')
        plt.axhline(y=0.5, color='gray', linestyle=':', label='Minimum Threshold (0.5)')
        plt.title("Walkforward Efficiency (WFE) Evolution")
        plt.legend()
        plt.show()

    def get_summary(self): # Returns statistical summary of all WFs tested
        summary = []
        for key, res in self.all_wf_results.items():
            summary.append({
                "WF_Config": key,
                "Total_PnL": res['total_pnl'],
                'Avg_WFE': res['wfe'],
                "Consistency": np.std([r['wfe'] for r in res['runs']])
            })
        return pl.DataFrame(summary).sort("Total_PnL", descending=True)
        





# def generate_mock_data(n_params=10, n_days=500):
#     all_raw_data = []
#     start_date = datetime(2022, 1, 1)
#     dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_days)]
    
#     for i in range(n_params):
#         # Gera retornos: PS_0 é bom, PS_4 é lixo, outros são neutros
#         mu = 0.001 if i == 0 else (-0.001 if i == n_params-1 else 0.0)
#         pnls = np.random.normal(mu, 0.01, n_days).tolist()
#         all_raw_data.append([dates, pnls])
        
#     return all_raw_data

# import itertools
# def run_full_grid_test(raw_data, is_values: List[int], os_values: List[int], step_ratio: float = 1.0):
#     """
#     Executa o Walkforward para todas as combinações possíveis de IS e OS.
#     step_ratio: define o step como proporção do OS (ex: 1.0 significa step = os_len)
#     """
#     # Gera todas as combinações possíveis: [(21, 6), (21, 16), (63, 6), (63, 16)...]
#     full_grid = []
#     for is_l, os_l in itertools.product(is_values, os_values):
#         step = int(os_l * step_ratio)
#         full_grid.append([is_l, os_l, step])
    
#     # Instancia a classe com a grade completa
#     wf = Walkforward(
#         raw_data=raw_data,
#         wfm=full_grid,
#         returns='best_wf_comb'
#     )
    
#     return wf

# # --- SETUP DO TESTE ---
# is_options = [21, 63, 168, 252] #[21, 42, 63, 84, 105, 126, 147, 168, 189, 210, 231, 252]
# os_options = [21, 63, 168, 252] #[21, 42, 63, 84, 105, 126, 147, 168, 189, 210, 231, 252]

# raw_mock = generate_mock_data(n_params=5, n_days=1500)

# wf_grid = run_full_grid_test(raw_mock, is_options, os_options)
# wf_grid.analyze()
# wf_grid.plot_heatmap()
# wf_grid.plot_advanced_heatmap(metric='wfe_sharpe')


# Se step < os_len pode acontecer de ter janelas com trades sobrepostos o que pode criar um vies
# para resolver isso foi modificado para usar a média dos resultados sobrespostos
# 1. Desenvolver as métricas do gem
# 2. Testar com resultados reais








