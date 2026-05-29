from dataclasses import dataclass, field
from SystemManager import SystemManager, SystemManagerParams
from typing import Optional, Callable, Dict, List
import polars as pl, numpy as np

@dataclass
class ModelSystemManagerParams(SystemManagerParams):
    model_hierarchy: dict = field(default_factory=lambda: {"order_by": 'highest', "metric": 'profit_perc'})
    rebalance_frequency: str = 'weekly'
    close_open_trades_on_rebalance: bool = False

class ModelSystemManager(SystemManager): # Manages portfolio's model hierarchy 
    def __init__(self, msm_params: ModelSystemManagerParams):
        super().__init__(msm_params) # SystemManager attributes init
        
        self.model_hierarchy = dict(msm_params.model_hierarchy)
        self.rebalance_frequency = msm_params.rebalance_frequency
        self.close_open_trades_on_rebalance = msm_params.close_open_trades_on_rebalance

#||=========================================================================================||

    def _default_pre_compute(self, global_assets, timeline, aggr_ret, indicator_pool, param_sets, manager_level_key) -> dict:
        # By Default doesn't calculate anything else, but can be used to prepare signals or other stuff != indicators
        return indicator_pool
                       
    def _default_rank(self, i, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key) -> Dict[str, float]:
        df_rets = pl.DataFrame(sim_data.get('both', {})).fill_null(0.0)
        if df_rets.is_empty() or df_rets.width < 1: return {}

        scores = {}
        corr_matrix = df_rets.corr() if df_rets.width > 1 else None

        for col in df_rets.columns:
            series = df_rets[col]
            std = series.std()
            sharpe = (series.mean() / std * np.sqrt(252)) if std > 0 else 0.0

            if corr_matrix is not None:
                avg_corr = (corr_matrix[col].sum() - 1) / (df_rets.width - 1)
            else:
                avg_corr = 0.0

            scores[col] = sharpe * (1 - avg_corr)
        
        return scores

    def _default_filter(self, i, step_dt, hierarchy: dict, indicator_pool: dict, scores: dict, port_returns: dict, key) -> List[str]:
        # Enables only top N asset/strat based on ranking
        top_n = getattr(self.params, 'top_n', 5)
        max_asset_per_strat_n = getattr(self.params, 'max_asset_per_strat_n', None)
        order_by = self.model_hierarchy['order_by']

        # Filters only those with score > 0 and takes N self.model_hierarchy['order_by']
        valid_scores = {k: v for k, v in scores.items() if v > 0}
        ranked_keys = sorted(valid_scores, key=valid_scores.get, reverse=True)[:top_n]

        for s_name, s_node in hierarchy.get('strats', {}).items():
            for a_name, a_node in s_node.get('assets', {}).items():
                item_key = f"{s_name}_{a_name}"

                if item_key in ranked_keys:
                    a_node['active'] = True
                    a_node['score'] = scores[item_key]
                else:
                    a_node['active'] = False
                    a_node['score'] = 0.0

        return hierarchy
    
    def _generate_internal_weights(self, i, step_dt, hierarchy: dict, scores: dict) -> dict:
        # Converts approved asset scores in percent weights that sum up to 1.0, later MM will use this to apply capital
        total_score = 0.0
        active_nodes = []

        for s_name, s_node in hierarchy.get("strats", {}).items():
            for a_name, a_node in s_node.get('assets', {}).items():
                if a_node.get('active', False):
                    total_score += a_node.get('score', 0.0)
                    active_nodes.append(a_node)

        # Distribute weigts proportionally
        for a_node in active_nodes:
            if total_score > 0:
                relative_weight = a_node.get('score', 0.0) / total_score
            else:
                relative_weight = 1.0 / len(active_nodes)

            # Applies weights on hierarchy
            l_factor = getattr(self.params, 'long_factor', 0.5)
            s_factor = getattr(self.params, 'short_factor', 0.5)

            if 'long' in a_node: a_node['long']['weight'] = relative_weight * l_factor 
            if 'short' in a_node: a_node['short']['weight'] = relative_weight * s_factor
            if 'both' in a_node: a_node['both']['weight'] = relative_weight

        return hierarchy
    

    def _default_rebalance(self, i, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key) -> List[str]:
        return hierarchy

    # ── Every Datetime [i] ───────────────────────────────────────────────
    
    def _default_main(self, i, step_dt, hierarchy: dict, indicator_pool: dict, port_returns: dict, key) -> dict:
        lookback = getattr(self.params, 'reb_lookback', 63)
        sim_data = self.get_data(key=key, lookback=lookback, data_type="aggr", side="both")

        # =========================================================================
        # 🟢 INÍCIO DO BLOCO DE DEBUG (Análise X/Y Quadrantes - Corrigido)
        # =========================================================================
        is_last_index = (i == len(self.portfolio.datetime_timeline) - 1)
    
        model_meta = self.portfolio.get_metadata_by_key(key)
        
        if model_meta and 'strats' in model_meta and sim_data and 'both' in sim_data and len(sim_data['both']['data']) > 1:
            
            print(f"\n{'='*90}")
            print(f"🔬 [MSM DEBUG] Análise de Quadrantes | Modelo: {key[1]} | Index: {i} | Janela: {lookback} barras")
            print(f"{'='*90}")

            # Converte o dicionário interno ("data" e "cols") do sim_data para DataFrame do Polars
            payload = sim_data['both']
            df_gran = pl.DataFrame(payload['data'], schema=payload['cols']).fill_null(0.0)
            
            # DIAGNÓSTICO: Printa as colunas reais para garantir o alinhamento
            if i % 2500 == 0 or is_last_index:
                print(f"🔍 [DIAGNÓSTICO] Colunas detectadas no sim_data: {df_gran.columns}")
            
            plot_points = []
            
            if df_gran.width > 0:
                strat_series = {}
                asset_metrics = {}

                # 1. PROCESSAMENTO MICRO (Nível Asset)
                for s_name, s_data in model_meta['strats'].items():
                    assets_in_strat = s_data.get('assets', [])
                    
                    # CORREÇÃO CRÍTICA: O Portfolio.py salva as colunas originais pelo nome do ativo (a_n),
                    # ou com o formato unificado. Varremos as duas possibilidades:
                    valid_cols = []
                    for a in assets_in_strat:
                        if a in df_gran.columns:
                            valid_cols.append(a)
                        elif f"{s_name}_{a}" in df_gran.columns:
                            valid_cols.append(f"{s_name}_{a}")

                    if not valid_cols: 
                        print(f"⚠️ [AVISO] Nenhuma coluna válida encontrada para a estratégia {s_name} (Ativos esperados: {assets_in_strat})")
                        continue

                    df_strat_assets = df_gran.select(valid_cols)
                    corr_matrix_asset = df_strat_assets.corr() if df_strat_assets.width > 1 else None
                    strat_total_pnl = np.zeros(df_gran.height)

                    for col in valid_cols:
                        series = df_strat_assets[col].to_numpy()
                        strat_total_pnl += series 
                        
                        asset_aggr = series.sum() 
                        std_val = series.std()
                        
                        # CORREÇÃO MATEMÁTICA: Proteção contra ativos que não operaram na janela (Std == 0)
                        if std_val == 0 or np.isnan(std_val):
                            vol_asset = 0.0
                            avg_corr_asset = 0.0
                        else:
                            vol_asset = std_val * np.sqrt(252)
                            if corr_matrix_asset is not None and col in corr_matrix_asset.columns:
                                corr_sum = corr_matrix_asset[col].sum() - 1
                                num_others = df_strat_assets.width - 1
                                avg_corr_asset = corr_sum / num_others if num_others > 0 else 0.0
                                if np.isnan(avg_corr_asset): avg_corr_asset = 0.0
                            else:
                                avg_corr_asset = 0.0

                        eixo_y = asset_aggr * (1 - avg_corr_asset)
                        if np.isnan(eixo_y): eixo_y = 0.0
                        
                        asset_metrics[col] = {
                            "aggr": asset_aggr, "vol": vol_asset, 
                            "corr": avg_corr_asset, "eixo_y": eixo_y,
                            "s_name": s_name
                        }

                    strat_series[s_name] = strat_total_pnl

                # 2. PROCESSAMENTO MACRO (Nível Strat)
                if strat_series:
                    df_strats = pl.DataFrame(strat_series)
                    corr_matrix_strat = df_strats.corr() if df_strats.width > 1 else None

                    print(f"\n{'Strat_Asset (Identificador)':<40} | {'Vol. Anual':<10} | {'Eixo X (Strat)':<15} | {'Eixo Y (Asset)':<15}")
                    print("-" * 90)

                    for col, metrics in asset_metrics.items():
                        s_name = metrics["s_name"]
                        strat_s = df_strats[s_name].to_numpy()
                        strat_aggr = strat_s.sum()
                        
                        if corr_matrix_strat is not None and s_name in corr_matrix_strat.columns:
                            corr_sum_strat = corr_matrix_strat[s_name].sum() - 1
                            num_others_strat = df_strats.width - 1
                            avg_corr_strat = corr_sum_strat / num_others_strat if num_others_strat > 0 else 0.0
                            if np.isnan(avg_corr_strat): avg_corr_strat = 0.0
                        else:
                            avg_corr_strat = 0.0

                        eixo_x = strat_aggr * (1 - avg_corr_strat)
                        if np.isnan(eixo_x): eixo_x = 0.0
                        
                        plot_points.append((col, eixo_x, metrics['eixo_y'], metrics['vol']))

                        print(f"{col:<40} | {metrics['vol']:>9.2%} | {eixo_x:>13.4f} | {metrics['eixo_y']:>13.4f}")
            print(f"{'='*90}\n")
            
            # 3. RENDERIZAÇÃO DO GRÁFICO (Apenas no final do backtest)
            if is_last_index and plot_points:
                import matplotlib.pyplot as plt
                
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(11, 8))
                fig.patch.set_facecolor('#0a0a0a')
                ax.set_facecolor('#0a0a0a')
                
                ax.set_title(f"Quadrantes de Força | Modelo: {key[1]}", color='white', pad=15, fontsize=14, fontweight='bold')
                ax.set_xlabel("Eixo X (Momento da Estratégia Ajustado por Corr)", color='#aaaaaa', labelpad=10)
                ax.set_ylabel("Eixo Y (Momento do Ativo Ajustado por Corr)", color='#aaaaaa', labelpad=10)
                
                ax.axhline(0, color='#555555', linestyle='--', linewidth=1.2, alpha=0.7)
                ax.axvline(0, color='#555555', linestyle='--', linewidth=1.2, alpha=0.7)
                
                for col_name, e_x, e_y, vol in plot_points:
                    # Limita o tamanho máximo e mínimo da bolha para evitar distorções visuais
                    bubble_size = clip_size = np.clip(vol * 3000, 60, 800)
                    ax.scatter(e_x, e_y, s=bubble_size, alpha=0.65, edgecolors='#ffffff', linewidth=0.8, cmap='viridis')
                    ax.annotate(col_name, (e_x, e_y), xytext=(7, 7), textcoords='offset points', color='#ffffff', fontsize=9, alpha=0.9)
                
                ax.grid(True, linestyle=':', alpha=0.15)
                ax.tick_params(colors='#888888', labelsize=10)
                plt.tight_layout()
                plt.show()
        else:
            if i == 0: 
                print(f" ⏳ [MSM DEBUG] Aguardando dados para Quadrantes do Modelo: {key[1]}...")
        # =========================================================================
        # 🔴 FIM DO BLOCO DE DEBUG
        # =========================================================================

        return hierarchy

#||=========================================================================================||








    """ # NOTE Não deletar abaixo, exemplo de MSM
        FinancialWisdom_Explosive_Stock_Asset_Rank_System

        - price < 100.0
        - market_cap < 10 Billion
        - Tech and Biotech Sectors (Sector Allocation Distribution)
        - Momentum Precedes Explosive Moves (Forecasts and Technical Indicators)
        - Weekly Timeframe

    """


