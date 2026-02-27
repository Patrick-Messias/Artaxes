import polars as pl, pandas as pd, numpy as np
from typing import List, Literal, Dict, Any, Union
import matplotlib.pyplot as plt, seaborn as sns
from datetime import datetime, timedelta

class Walkforward:
    def __init__(
        self,
        wfm: List[List[int]],
        raw_data: List[List[Union[List[str], List[float]]]] = None,
        wf_res: str='weekly',
        wf_metric: List[Any] = [1, 'highest', 'pnl'],
        wfm_wf_selection: List[Any] = [1, 'highest', 'wfe'],
        returns: Literal['top', 'best_wf_comb', 'all'] = 'top',
    ):
        self.wfm_configs = wfm
        self.wf_metric = wf_metric
        self.wfm_wf_selection = wfm_wf_selection
        self.returns_mode = returns
        if raw_data is not None:
            self.df = self._consolidate_to_wide_df(raw_data) 
        else:
            self.df = pl.DataFrame() # Empty Placeholder
        self.all_wf_results = {}

    def _consolidate_to_wide_df(self, raw_data):
        dfs_to_join = []

        for i, (dts, pnls) in enumerate(raw_data):
            # Groups PnL by Datetime
            temp_df = pl.DataFrame({"datetime": dts, f"ps_{i}": pnls})
            temp_df = temp_df.with_columns(pl.col("datetime").str.to_date())
            temp_df = temp_df.group_by("datetime").agg(pl.col(f"ps_{i}").sum())
            dfs_to_join.append(temp_df)

        full_df = dfs_to_join[0]
        for next_df in dfs_to_join[1:]:
            full_df = full_df.join(
                next_df, 
                on="datetime", 
                how="full",
                coalesce=True
            )
        return full_df.sort("datetime").fill_null(0.0)

    def _calculate_fitness(self, df_slice: pl.DataFrame):
        metric = self.wf_metric[2]
        if metric == 'pnl':
            return df_slice.sum()
        elif metric == 'sharpe':
            return df_slice.mean() / (df_slice.std() + 1e-9)
        else:
            print(f"Unknown metric '{metric}', defaulting to PnL")
            return df_slice.sum() # Default to sum if unknown metric

    def run_wf(self, is_len, os_len, step):
        total_rows = len(self.df)
        chunks = []

        for start in range(0, total_rows - is_len - os_len + 1, step):
            is_idx = (start, start + is_len)
            os_idx = (is_idx[1], is_idx[1] + os_len)

            is_df = self.df.slice(is_idx[0], is_len).drop("datetime")
            fitness_df = self._calculate_fitness(is_df)

            # Selects best param_set and converts to Series to use arg_max/min
            fitness_series = fitness_df.row(0, named=True)

            # Logic to find the name of the columns (param_set)
            if self.wf_metric[1] == 'highest':
                best_ps = max(fitness_series, key=lambda k: fitness_series[k])
            else:
                best_ps = min(fitness_series, key=lambda k: fitness_series[k])

            # Collects out of sample performance
            os_data = self.df.slice(os_idx[0], os_len).select(["datetime", best_ps])
            os_returns = os_data.select(best_ps).to_series()

            # Calculates WFE
            os_perf = os_returns.sum()
            os_sharpe = (os_returns.mean() / (os_returns.std() + 1e-9)) * np.sqrt(252)

            is_perf = fitness_series[best_ps]
            wfe = (os_perf / os_len) / (is_perf / is_len) if is_perf != 0 else 0

            chunks.append({
                "os_curve": os_data.rename({best_ps: "pnl"}),
                "wfe": wfe,
                "metrics": {
                    "is_pnl": is_perf,
                    "os_pnl": os_perf,
                    "os_sharpe": os_sharpe,
                    "os_ev": os_returns.mean() # Valor Esperado por trade/dia
                }
            })
        return chunks

    def analyze(self):
        for config in self.wfm_configs:
            is_l, os_l, stp = config
            wf_key = f"IS{is_l}_OS{os_l}"

            runs = self.run_wf(is_l, os_l, stp)
            if not runs: continue
        
            full_oos_curve = pl.concat([r["os_curve"] for r in runs]).sort("datetime")

            # Calculates Max Drawdown (Not underwater, but has the resolution of pnl curve)
            cum_pnl = full_oos_curve["pnl"].cum_sum()
            running_max = cum_pnl.cum_max()
            drawdown = running_max - cum_pnl
            max_dd = drawdown.max()

            # WFE Sharpe Ratio
            wfes = [r["wfe"] for r in runs]
            wfe_sharpe = np.mean(wfes) / (np.std(wfes) + 1e-9)

            self.all_wf_results[wf_key] = {
                "curve": full_oos_curve,
                "total_pnl": full_oos_curve["pnl"].sum(),
                "max_dd": max_dd,
                "avg_sharpe": np.mean([r["metrics"]["os_sharpe"] for r in runs]),
                "avg_ev": np.mean([r["metrics"]["os_ev"] for r in runs]),
                "wfe_mean": np.mean(wfes),
                "wfe_sharpe": wfe_sharpe,
                "runs": runs,
                "config": config
            }
        return self._finalize()
    
    def _finalize(self):
        if self.returns_mode == 'top':
            best_wf = max(self.all_wf_results, key=lambda k: self.all_wf_results[k]['total_pnl'])
            return self.all_wf_results[best_wf]
        elif self.returns_mode == 'best_wf_comb':
            return self._get_best_combination_curve()
        
        return self.all_wf_results


    def _get_best_combination_curve(self):
        combination_oos_data = []

        # Aligns all OOS curves into a single DataFrame for comparison
        for wf_key, result in self.all_wf_results.items():
            curve = result['curve'].rename({"pnl": wf_key})
            combination_oos_data.append(curve)

        # DataFrame with all WF options side by side
        matrix_df = combination_oos_data[0]
        for next_df in combination_oos_data[1:]:
            matrix_df = matrix_df.join(next_df, on="datetime", how="full", coalesce=True)

        # Makes shure doen't exist duplicated data that breaks .item()
        matrix_df = matrix_df.group_by("datetime").agg(pl.all().mean()).sort("datetime").fill_null(0.0) #matrix_df = matrix_df.sort("datetime").unique(subset=["datetime"]).fill_null(0.0)

        # Creates a ranking with WFE to find best combination for each cronological window
        final_data = []
        all_dates = matrix_df["datetime"].unique().sort()
        current_best_wf = list(self.all_wf_results.keys())[0]

        for dt in all_dates: # Updates best WF combination based on already closed windows before data 'dt'
            best_val = -float('inf') if self.wfm_wf_selection[1] == 'highest' else float('inf')

            for wf_key, content in self.all_wf_results.items(): # Filters runs (windows) that finished before 'dt'
                past_runs = [r for r in content['runs'] if r['os_curve']['datetime'].max() < dt]

                if not past_runs: continue

                # Calculates selection metric (WFE of historical average)
                if self.wfm_wf_selection[2] == 'wfe':
                    current_metric = np.mean([r['wfe'] for r in past_runs])
                else: # pnl
                    current_metric = sum([r['metrics']['os_pnl'] for r in past_runs])

                if self.wfm_wf_selection[1] == 'highest':
                    if current_metric > best_val:
                        best_val = current_metric
                        current_best_wf = wf_key
                else:
                    if current_metric < best_val:
                        best_val = current_metric
                        current_best_wf = wf_key

            # Takes chosen WF daily PnL
            day_slice = matrix_df.filter(pl.col("datetime") == dt).select(current_best_wf)
            pnl_val = day_slice.row(0)[0] if day_slice.height > 0 else 0.0
            final_data.append({
                "datetime": dt,
                "pnl": pnl_val,
                "selected_wf": current_best_wf
            })
        return pl.DataFrame(final_data)
    
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








