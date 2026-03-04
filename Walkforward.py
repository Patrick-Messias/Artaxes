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
        wfm_is_always_higher_or_equal_to_oos: bool=True,
        matrix_resolution: Literal['daily', 'weekly', 'monthly'] = 'weekly',
        time_mode: Literal['trade_days', 'calendar_days'] = 'calendar_days',

        # Parset Selection
        is_metric: Literal['pnl', 'sharpe', 'pnl_dd'] = 'pnl',
        is_top_n: int=1,
        is_logic: Literal['highest', 'mode'] = 'highest',
        is_order: SortOrder = 'des',

        # Final WF Seletion from WFM
        wf_selection_metric: Literal['pnl', 'pnl_dd', 'pnl_sharpe', 'wfe', 'wfe_sharpe'] = 'pnl',
        wf_selection_analysis_radius_n: int=2,
        wf_selection_logic: Literal['highest', 'highest_stable', 'stable'] = 'highest',
        wf_returns_mode: Literal['selected', 'all'] = 'selected'
    ):
        # A matrix já vem pronta do seu _process_wfm_to_polars no Operation
        self.matrix = matrix 
        self.wfm_configs = wfm_configs
        self.wfm_is_always_higher_or_equal_to_oos = wfm_is_always_higher_or_equal_to_oos
        self.matrix_resolution = matrix_resolution
        self.time_mode = time_mode

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
                (pl.col(c).sum() / (pl.col(c).cum_sum().max() - pl.col(c).cum_sum().min() + 1e-6)).alias(c)
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
        all_oos_returns = []

        start_idx = 0
        current_is_end_idx = is_periods

        _WFE_MIN_DENOMINATOR = 1e-4  # Magnitude mínima do IS para WFE ser válido

        while current_is_end_idx < total_rows:
            current_oos_end_idx = min(current_is_end_idx + os_periods, total_rows)
            os_len = current_oos_end_idx - current_is_end_idx

            is_data = df.slice(start_idx, is_periods)
            os_data = df.slice(current_is_end_idx, os_len)
            
            if is_data.is_empty() or os_data.is_empty(): break

            best_ps_id = self._find_best_parameter(is_data)
            if best_ps_id is not None:
                os_returns = os_data[best_ps_id].to_numpy()
                all_oos_returns.extend(os_returns.tolist())

                # Cálculo de WFE (Walkforward Efficiency) - normalizado por período
                is_total_pnl = float(is_data[best_ps_id].sum())
                os_total_pnl = float(os_returns.sum())

                if abs(is_total_pnl) < _WFE_MIN_DENOMINATOR:
                    # IS perto de zero: denominador instável, janela excluída da média
                    wfe = float('nan')
                elif is_total_pnl < 0:
                    # IS negativo: razão sem interpretação válida (seleção em período perdedor)
                    wfe = 0.0
                else:
                    # Normaliza pelo comprimento real de cada janela para comparar configs diferentes
                    is_avg = is_total_pnl / is_periods
                    os_avg = os_total_pnl / os_len
                    wfe = os_avg / is_avg
                    # Clamp por janela individual — protege a média agregada de outliers
                    wfe = float(np.clip(wfe, -2.0, 2.0))

                # Sharpe OOS Simples
                std_val = os_returns.std()
                os_avg_pnl = os_returns.mean()
                os_sharpe = (os_avg_pnl / (std_val + 1e-9)) * np.sqrt(252)

                runs.append({
                    "is_df": is_data,
                    "os_curve": os_data.select(["ts", best_ps_id]).rename({best_ps_id: "pnl"}),
                    "best_param": best_ps_id,
                    "wfe": float(wfe) if not np.isnan(wfe) else float('nan'),
                    "metrics": {
                        "os_sharpe": float(os_sharpe), 
                        "is_metric_val": float(is_data[best_ps_id].sum())
                    }
                })
            
            start_idx += step_periods
            current_is_end_idx += step_periods
            if start_idx + is_periods > total_rows: break

        return {"windows": runs, "cumulative_oos": np.cumsum(all_oos_returns).tolist()}


    def analyze(self, is_always_higher_or_equal_to_oos=True):
        if self.matrix is None or self.matrix.is_empty():
            print("      > [Error] Matrix is empty")
            return None

        self.all_wf_results = {} 
        working_df = self.matrix.clone()
        
        # Ajuste de Resolução
        if self.time_mode == 'calendar_days':
            working_df = working_df.upsample(time_column="ts", every="1d").fill_null(0)

        if self.matrix_resolution == 'weekly':
            working_df = working_df.group_by_dynamic("ts", every="1w", closed="left").agg(pl.all().exclude("ts").sum())
        elif self.matrix_resolution == 'monthly':
            working_df = working_df.group_by_dynamic("ts", every="1mo", closed="left").agg(pl.all().exclude("ts").sum())

        self._temp_df = working_df.sort("ts")
        total_rows = len(self._temp_df)

        # Filtro de Robustez: IS >= OS
        configs_to_run = self.wfm_configs
        if is_always_higher_or_equal_to_oos:
            configs_to_run = [c for c in self.wfm_configs if c[0] >= c[1]]
            print(f"      > [WFM] Running {len(configs_to_run)} robust configurations (IS >= OS).")

        for config in configs_to_run:
            is_l, os_l, stp = config
            if is_l + os_l > total_rows: continue

            wf_key = f"IS{is_l}_OS{os_l}_ST{stp}"
            wf_data = self.run_wf(is_l, os_l, stp)
            
            if not wf_data or not wf_data["windows"]: continue
            runs = wf_data["windows"]
        
            full_oos_curve = pl.concat([r["os_curve"] for r in runs]).sort("ts")
            equity_curve = full_oos_curve["pnl"].cum_sum()
            
            total_pnl = float(full_oos_curve["pnl"].sum() or 0.0)
            max_dd = float((equity_curve.cum_max() - equity_curve).max() or 0.0)
            pnl_dd = float(total_pnl / (max_dd + 1e-9) or 0.0)
            
            # Agrega WFE excluindo janelas inválidas (nan) antes da média
            raw_wfes = [r['wfe'] for r in runs]
            valid_wfes = [v for v in raw_wfes if not np.isnan(v)]

            if valid_wfes:
                avg_wfe = float(np.mean(valid_wfes))
                # Clamp final na média como segunda camada de defesa
                avg_wfe = float(np.clip(avg_wfe, -2.0, 2.0))
                wfe_std = float(np.std(valid_wfes)) if len(valid_wfes) > 1 else 1.0
            else:
                avg_wfe = 0.0
                wfe_std = 1.0
            
            pnl_sharpe = float(np.mean([r['metrics']['os_sharpe'] for r in runs]))
            
            windows_metadata = []
            current_global_idx = 0
            for r in runs:
                pnl_data = r["os_curve"]["pnl"].to_numpy()
                n_points = len(pnl_data)
                global_indices = np.arange(current_global_idx, current_global_idx + n_points)
                prev_cum_pnl = equity_curve[current_global_idx - 1] if current_global_idx > 0 else 0.0
                window_equity = np.cumsum(pnl_data) + prev_cum_pnl

                windows_metadata.append({
                    'is_start': r['is_df']['ts'].min(), 'is_end': r['is_df']['ts'].max(),
                    'os_start': r['os_curve']['ts'].min(), 'os_end': r['os_curve']['ts'].max(),
                    'best_param': r.get('best_param', 'N/A'),
                    'metric_val': r['metrics'].get('is_metric_val', 0.0),
                    'global_x': global_indices.tolist(), 'global_y': window_equity.tolist()
                })
                current_global_idx += n_points

            self.all_wf_results[wf_key] = {
                "config": config,
                "equity": equity_curve.to_list(),
                "total_pnl": total_pnl,
                "max_dd": max_dd,
                "pnl_dd": pnl_dd,
                "pnl_sharpe": pnl_sharpe,
                "wfe": avg_wfe,
                "wfe_sharpe": avg_wfe / (wfe_std + 1e-9),
                "runs": runs,
                "windows": windows_metadata
            }
        
        return self._finalize()

    def _finalize(self):
        if not self.all_wf_results:
            print("      > [Warning] No Walkforward results to finalize")
            return None
        
        _metric_alias = {
            'pnl': 'total_pnl',
        }
        metric_key = _metric_alias.get(self.wf_selection_metric, self.wf_selection_metric)

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
    def plot_heatmap(self, metric='wfe'):
        plt.style.use('dark_background')
        
        data = []
        for res in self.all_wf_results.values():
            data.append({
                "IS": res['config'][0], 
                "OS": res['config'][1], 
                "Val": res[metric]
            })

        if not data:
            print("      > [Error] No data to plot heatmap.")
            return

        df_plot = pd.DataFrame(data).pivot(index="IS", columns="OS", values="Val")
        # Garantir que o eixo IS fique em ordem decrescente
        df_plot = df_plot.sort_index(ascending=False)

        plt.figure(figsize=(12, 8))
        sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#0a0a0a", "figure.facecolor": "#0a0a0a"})

        # mask detecta onde os dados são nulos (combinações IS < OS filtradas)
        mask = df_plot.isnull()

        ax = sns.heatmap(
            df_plot, 
            annot=True, 
            fmt=".2f", 
            cmap="RdYlGn", 
            center=0 if metric != 'max_dd' else None,
            mask=mask, # Isso mantém a parte filtrada escura/vazia
            annot_kws={"size": 10, "weight": "bold"},
            linewidths=.5, 
            linecolor='#333333',
            cbar_kws={'label': metric.upper()}
        )

        plt.title(f"WFM Matrix - {metric.upper()} Analysis", color='white', fontsize=14, pad=20)
        plt.xlabel("Out-of-Sample (OS)", color='white')
        plt.ylabel("In-Sample (IS)", color='white')
        plt.xticks(color='white')
        plt.yticks(color='white')
        plt.show()

    def plot_advanced_heatmap(self, metric: Literal['total_pnl', 'max_dd', 'pnl_sharpe', 'wfe_sharpe'] = 'total_pnl'):
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
        
    def plot_timeline(self, wf_result: Dict, show_is=True, show_os=True, show_metrics=True):
        if not wf_result or 'windows' not in wf_result or not wf_result['windows']:
            print("      > [Error] No windows data to plot timeline.")
            return

        import matplotlib.dates as mdates
        from matplotlib.patches import Patch
        
        plt.style.use('dark_background')
        windows = wf_result['windows']
        config = wf_result['config']
        
        color_is = '#004d4d' # Azul Petróleo
        color_os = '#4169E1' # Azul Royal
        
        # Aumenta a altura proporcionalmente ao número de janelas
        fig, ax = plt.subplots(figsize=(14, len(windows) * 0.4 + 2))
        fig.patch.set_facecolor('#0a0a0a')
        ax.set_facecolor('#0a0a0a')

        use_index = self.time_mode == 'trade_days'
        
        # Rastreadores para garantir que o eixo X cubra tudo
        x_min = float('inf')
        x_max = float('-inf')

        for i, win in enumerate(windows):
            if use_index:
                # Lógica para TRADE_DAYS (Índices sequenciais)
                is_start_val = i * config[2]
                is_width = config[0]
                os_start_val = is_start_val + is_width
                os_width = config[1]
                
                current_end = os_start_val + os_width
                current_start = is_start_val
            else:
                # Lógica para CALENDAR_DAYS (Datas reais)
                is_start_val = mdates.date2num(win['is_start'])
                is_end_val = mdates.date2num(win['is_end'])
                os_start_val = mdates.date2num(win['os_start'])
                os_end_val = mdates.date2num(win['os_end'])
                
                is_width = is_end_val - is_start_val
                os_width = os_end_val - os_start_val

                # Correção para garantir visibilidade mínima
                is_width = max(is_width, 0.1)
                os_width = max(os_width, 0.1)
                
                current_start = is_start_val
                current_end = os_end_val

            # Atualiza os limites globais do gráfico
            x_min = min(x_min, current_start)
            x_max = max(x_max, current_end)

            # Plotagem das barras In-Sample
            if show_is:
                ax.broken_barh([(is_start_val, is_width)], (i - 0.4, 0.8), 
                               facecolors=color_is, alpha=0.5, edgecolor='#1a1a1a', linewidth=0.5)
            
            # Plotagem das barras Out-of-Sample
            if show_os:
                ax.broken_barh([(os_start_val, os_width)], (i - 0.4, 0.8), 
                               facecolors=color_os, alpha=1.0, edgecolor='white', linewidth=1.0)

            # Métricas ao lado de cada janela
            if show_metrics:
                text_x = os_start_val + os_width
                ax.text(text_x, i, f"  {win['best_param']} ({win['metric_val']:.2f})", 
                        va='center', color='white', fontsize=8, alpha=0.8)

        # --- AJUSTE DOS LIMITES E FORMATAÇÃO ---
        
        # Adiciona 10% de margem à direita para o texto das métricas não ser cortado
        margin = (x_max - x_min) * 0.10
        ax.set_xlim(x_min, x_max + margin)
        
        # Garante que o eixo Y mostre todas as janelas (incluindo a última no topo)
        ax.set_ylim(-1, len(windows))

        if not use_index:
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45, ha='right', color='#888888')
        else:
            ax.set_xlabel("Sequential Periods / Trades", color='#888888')

        ax.set_yticks(range(len(windows)))
        ax.set_yticklabels([f"W {i+1}" for i in range(len(windows))], color='#888888')
        
        title_str = f"Walkforward Timeline | {self.time_mode.upper()} | IS:{config[0]} OS:{config[1]} ST:{config[2]}"
        ax.set_title(title_str, color='white', loc='left', pad=25, fontsize=12)
        
        ax.grid(True, axis='x', linestyle='--', alpha=0.1)
        
        # Legenda
        legend_elements = [
            Patch(facecolor=color_is, label='In-Sample (Train)'), 
            Patch(facecolor=color_os, label='Out-of-Sample (Live)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', frameon=False, 
                  bbox_to_anchor=(1, 1.12), ncol=2, fontsize=9)

        plt.tight_layout()
        plt.show()

    def plot_oos_curves(self, wf_result: Dict = None):
        if not self.all_wf_results:
            print("      > [Error] No results found.")
            return

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#0a0a0a')
        ax.set_facecolor('#0a0a0a')

        # Se wf_result foi passado (da Operation.py), tentamos extrair a chave dele
        selected_key = None
        if wf_result and 'config' in wf_result:
            c = wf_result['config']
            selected_key = f"IS{c[0]}_OS{c[1]}_ST{c[2]}"
        else:
            selected_key = max(self.all_wf_results, key=lambda k: self.all_wf_results[k]['total_pnl'])

        for wf_key, data in self.all_wf_results.items():
            if wf_key == selected_key:
                ax.plot(data['equity'], color='#4169E1', lw=2.5, label=f"Selected: {wf_key}", zorder=10)
            else:
                ax.plot(data['equity'], color='#C3C3C3', lw=1, alpha=0.2, zorder=1)

        ax.set_title("Walkforward Matrix Evolution", color='white', loc='left')
        ax.legend(frameon=False, fontsize='small')
        plt.show()




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








