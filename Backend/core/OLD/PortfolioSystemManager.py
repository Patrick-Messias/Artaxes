from dataclasses import dataclass, field
from SystemManager import SystemManager, SystemManagerParams
from typing import List, Optional, Dict, Literal, Callable, Union
#from Backend.core import Asset
import polars as pl, numpy as np

@dataclass
class PortfolioSystemManagerParams(SystemManagerParams):
    model_hierarchy: Dict = field(default_factory=lambda: {
        "order_by": "highest",
        "metric":   "pnl"
    })
    max_active_models: Optional[int] = None # Max number of active models in the portfolio at any given time (if None, no limit)

    # Rebalancing
    reb_metric: Literal["pnl", "pnl_dd", "sharpe"] = "pnl" # Metric used for performance-based rebalancing (if reb_method == "performance")
    reb_method: Literal["fixed", "equal_weight", "risk_parity", "performance"] = "fixed"
    reb_deviation_func: Optional[Dict[str, Callable]] = None # Only rebalance if (ex: Portfolio std deviated "x" std from mean)
    reb_closes_open_trades_on_rebalance: bool = False # NOTE add this only to StratSystemManager

class PortfolioSystemManager(SystemManager): # Manages portfolio's model hierarchy 
    def __init__(self, psm_params: PortfolioSystemManagerParams):
        super().__init__(psm_params)

        self.reb_metric                         = getattr(psm_params, 'reb_metric', 'pnl')
        self.model_hierarchy                    = dict(psm_params.model_hierarchy)
        self.max_active_models                  = psm_params.max_active_models
        self.reb_method                         = getattr(psm_params, 'reb_method', 'fixed')
        self.reb_closes_open_trades_on_rebalance = psm_params.reb_closes_open_trades_on_rebalance

#||=========================================================================================||

    def _default_pre_compute(self, global_assets, timeline, aggr_ret, indicator_pool, param_sets, manager_level_key) -> dict:
        # By Default doesn't calculate anything else, but can be used to prepare signals or other stuff != indicators
        return indicator_pool
                       
    # ── Every Datetime [i] ───────────────────────────────────────────────
    
    def _default_rank(self, i, step_dt, hierarchy, indicator_pool, aggr_data, port_returns, key) -> dict:
        import numpy as np

        for sd in ["both", "long", "short"]:
            if sd not in aggr_data: continue

            for m_key, m_info in self.portfolio.iter_hierarchy(target_level="models"):
                # O nome da coluna no portf_aggr é f"{op}_{model}"
                col_name = f"{m_key[0]}_{m_key[1]}"
                if col_name in aggr_data[sd].keys():
                    pnl = np.sum(aggr_data[sd][col_name])
                    m_info[sd]['score'] = 1.0 if pnl > 0.0 else 0.0
                else: 
                    continue

        return hierarchy

    def _default_filter(self, i, step_dt, hierarchy, indicator_pool, aggr_data, port_returns, key) -> dict:
        for m_key, m_info in self.portfolio.iter_hierarchy(target_level="models"):
            side_config = m_info.get("side", "both")
            separate_ls = m_info.get("separate_ls", False)
            
            sides = ["long", "short"] if (side_config == "both" and separate_ls) \
                    else [side_config if side_config != "both" else "both"]

            is_active = False
            for sd in sides:
                score = m_info[sd].get('score', 0.0)
                # Lógica de filtro (exemplo: score mínimo)
                if score > -0.1: 
                    is_active = True
            
            m_info['active'] = is_active
            
        return hierarchy

    def _default_rebalance(self, i, step_dt, hierarchy, indicator_pool, aggr_data, port_returns, key) -> dict:
        import numpy as np, polars as pl
        from indicators.HRP import HRP

        lookback = self.reb_lookback

        # 1. Identifica modelos ativos e limpa pesos
        active_models = []
        for m_key, m_info in self.portfolio.iter_hierarchy(target_level="models"):
            for s in ["both", "long", "short"]: 
                if s in m_info: m_info[s]['weight'] = 0.0
            if m_info.get('active', True):
                active_models.append(m_key)

        if len(active_models) < 2: 
            return hierarchy

        # 2. Processa cada via (side)
        for sd in ["both", "long", "short"]:
            if sd not in aggr_data: continue
            
            matrix_data = {}
            target_col_names = []
            
            for m_key in active_models:
                col_name = f"{m_key[0]}_{m_key[1]}"
                
                if col_name in aggr_data[sd]:
                    # --- BLINDAGEM 1: Conversão forçada no NumPy ---
                    # Garante que o dado saia da lista Python como Float64 real
                    raw_val = aggr_data[sd][col_name]
                    returns = np.array(raw_val, dtype=np.float64).flatten()
                    
                    # Ignora se for tudo NaN ou se a série for muito curta para correlação
                    if len(returns) > 1 and not np.all(np.isnan(returns)):
                        # Substitui Infs/NaNs por 0.0 para não quebrar a matriz de covariância
                        matrix_data[col_name] = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
                        target_col_names.append((m_key, col_name))

            if len(matrix_data) < 2: continue

            # 3. Cálculo do HRP com Sanitização de Schema
            try:
                # --- BLINDAGEM 2: Schema Explícito no Polars ---
                # Definir o schema impede que o Polars tente "adivinhar" que é String
                explicit_schema = {name: pl.Float64 for name in matrix_data.keys()}
                
                df_hrp = (
                    pl.DataFrame(matrix_data, schema=explicit_schema)
                    .fill_null(0.0)
                    .fill_nan(0.0)
                )

                # --- BLINDAGEM 3: Filtro de segurança pré-HRP ---
                # Garante que nenhuma coluna String "escapou" para o cálculo
                df_hrp = df_hrp.select(pl.col(pl.Float64))
                if df_hrp.width < 2: continue

                hrp_engine = HRP()
                weights = hrp_engine._calculate_logic(df_hrp)
      
                # 4. Distribuição dos pesos
                for m_key, col_name in target_col_names:
                    if col_name in weights:
                        hierarchy[m_key][sd]['weight'] = float(weights[col_name]) 
                        if i % 1000 == 0 or i == 55394:
                            print(f"hierarchy[{m_key}][{sd}]['weight']: {hierarchy[m_key][sd]['weight']}")
                        
            except Exception as e:
                # O erro 55394 agora será capturado aqui sem travar o backtest
                if i % 1000 == 0 or i == 55394: # Evita spam de log, mas foca no erro
                    print(f"    < [PSM Rebalance] Side '{sd}' at idx {i} skip: {e}")

        return hierarchy


    def _default_main(self, i, step_dt, hierarchy: dict, indicator_pool: dict, port_returns: dict, key) -> bool:
        
        # Default uses aggr of models for Portfolio Level
        port_key = (self.portfolio.name,)
        port_aggr_data = self.get_data(key=port_key, lookback=self.reb_lookback, data_type="aggr", side=["both", "long", "short"])

        if not port_aggr_data:
            print("     < [PSM._default_main] No aggr data found")
            return hierarchy

        # 2. Roda o Rank (Vetorizado sobre a matriz global)
        hierarchy = self._default_rank(i, step_dt, hierarchy, indicator_pool, port_aggr_data, port_returns, port_key)

        # 3. Roda o Filter (Decide quem fica ativo baseado nos scores)
        hierarchy = self._default_filter(i, step_dt, hierarchy, indicator_pool, port_aggr_data, port_returns, port_key)

        # 4. Roda o Rebalance (Calcula pesos para os que sobreviveram)
        hierarchy = self._default_rebalance(i, step_dt, hierarchy, indicator_pool, port_aggr_data, port_returns, port_key)

        return hierarchy
   
#||=========================================================================================||

    """ Dt execution framework

    1. Check current tradable Models
    -> REBALANCE

    2. New Rank generated with updated data
    3. Needs to remove any Models? if yes then close or keep positions open by SM rules? != MM rules
    4. Needs to add any Models? 
    

    """


















