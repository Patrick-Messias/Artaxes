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
        # 🟢 INÍCIO DO BLOCO DE DEBUG (Análise X/Y Quadrantes)
        # =========================================================================
        if i % 500 == 0 or i == len(self.portfolio.datetime_timeline) - 1: 
            print(f"\n{'='*75}")
            print(f"🔬 [MSM DEBUG] Análise de Quadrantes | Modelo: {key[1]} | Index: {i}")
            print(f"{'='*75}")

            # CORREÇÃO 2: Acessando a função de metadados via self.portfolio
            model_meta = self.portfolio.get_metadata_by_key(key)
            
            if model_meta and 'strats' in model_meta and sim_data and 'both' in sim_data:
                df_gran = pl.DataFrame(sim_data['both']).fill_null(0.0)
                
                if df_gran.width > 0:
                    strat_series = {}
                    asset_metrics = {}

                    # 1. PROCESSAMENTO MICRO (Nível Asset)
                    for s_name, s_data in model_meta['strats'].items():
                        assets_in_strat = s_data.get('assets', [])
                        valid_cols = [f"{s_name}_{a}" for a in assets_in_strat if f"{s_name}_{a}" in df_gran.columns]
                        
                        if not valid_cols: continue

                        # Cria DataFrame isolado só com os ativos dessa estratégia
                        df_strat_assets = df_gran.select(valid_cols)
                        
                        # Correlação Micro (Entre ativos da mesma strat)
                        corr_matrix_asset = df_strat_assets.corr() if df_strat_assets.width > 1 else None

                        strat_total_pnl = np.zeros(df_gran.height)

                        for col in valid_cols:
                            series = df_strat_assets[col].to_numpy()
                            strat_total_pnl += series # Soma para construir a curva agregada da Strat
                            
                            asset_aggr = series.sum() # Retorno acumulado (Eficiência base)
                            vol_asset = series.std() * np.sqrt(252) # Volatilidade anualizada
                            
                            if corr_matrix_asset is not None:
                                avg_corr_asset = (corr_matrix_asset[col].sum() - 1) / (df_strat_assets.width - 1)
                            else:
                                avg_corr_asset = 0.0 # Se só tem 1 ativo, não há correlação para penalizar

                            # 🟢 EIXO Y: Eficiência isolada do ativo
                            eixo_y = asset_aggr * (1 - avg_corr_asset)
                            
                            asset_metrics[col] = {
                                "aggr": asset_aggr, "vol": vol_asset, 
                                "corr": avg_corr_asset, "eixo_y": eixo_y
                            }

                        # Salva a série agregada da Estratégia para ser usada no Nível Macro
                        strat_series[s_name] = strat_total_pnl

                    # 2. PROCESSAMENTO MACRO (Nível Strat)
                    if strat_series:
                        df_strats = pl.DataFrame(strat_series)
                        corr_matrix_strat = df_strats.corr() if df_strats.width > 1 else None

                        print(f"{'Strat_Asset':<25} | {'Vol. Anual':<10} | {'Eixo X (Strat)':<15} | {'Eixo Y (Asset)':<15}")
                        print("-" * 75)

                        for col, metrics in asset_metrics.items():
                            s_name = col.split("_")[0] # Identifica a qual strat o ativo pertence
                            
                            # Puxa os dados da Strat agregada
                            strat_s = df_strats[s_name].to_numpy()
                            strat_aggr = strat_s.sum()
                            
                            if corr_matrix_strat is not None:
                                avg_corr_strat = (corr_matrix_strat[s_name].sum() - 1) / (df_strats.width - 1)
                            else:
                                avg_corr_strat = 0.0

                            # 🟢 EIXO X: Eficiência global da estratégia no modelo
                            eixo_x = strat_aggr * (1 - avg_corr_strat)

                            print(f"{col:<25} | {metrics['vol']:>9.2%} | {eixo_x:>13.4f} | {metrics['eixo_y']:>13.4f}")
            else:
                print("⚠️ Metadados ou dados de simulação insuficientes para calcular os Quadrantes de DEBUG.")
            print(f"{'='*75}\n")
        # =========================================================================
        # 🔴 FIM DO BLOCO DE DEBUG
        # =========================================================================

        # if self.rebalance(i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key):
        #     scores = self.rank(i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)
        #     hierarchy = self.filter(i, step_dt, hierarchy, indicator_pool, scores, port_returns, key)
            # if hasattr(self, '_generate_internal_weights'):
            #     hierarchy = self._generate_internal_weights(i, step_dt, hierarchy, scores)

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


