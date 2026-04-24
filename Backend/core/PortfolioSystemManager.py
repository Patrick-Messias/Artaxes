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

        self.reb_metric                         = psm_params.reb_metric
        self.model_hierarchy                    = dict(psm_params.model_hierarchy)
        self.max_active_models                  = psm_params.max_active_models
        self.reb_method                         = psm_params.reb_method
        self.reb_closes_open_trades_on_rebalance = psm_params.reb_closes_open_trades_on_rebalance

#||=========================================================================================||

    def _default_pre_compute(self, global_assets, timeline, aggr_ret, indicator_pool, param_sets, manager_level_key) -> dict:
        # By Default doesn't calculate anything else, but can be used to prepare signals or other stuff != indicators
        return indicator_pool
                       
    # ── Every Datetime [i] ───────────────────────────────────────────────
    
    def _default_rank(self, i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key) -> dict:
        # Ranks models by metric defined in model_hierarchy. Returns dict[model_name: score]
        for m_key in hierarchy['models'].keys():
            m_pnl = sim_data.get(m_key, {}).get('both', {}).get('data')
            if m_pnl is not None:
                hierarchy['models'][m_key]['score'] = float(m_pnl[i])
        return hierarchy

    def _default_filter(self, i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key) -> dict:
        # Disables models that don't pass the filter function. Returns dict with 'active' field updated

        for m_key in hierarchy['models'].keys():
            score = hierarchy['models'][m_key].get('score', 0)
            
            # Exemplo: Desativa se o score for negativo
            if score < 0:
                hierarchy['models'][m_key]['active'] = False
            
        return hierarchy

    def _default_rebalance(self, i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key) -> dict:
        """
        Realiza o rebalanceamento via Hierarchical Risk Parity (HRP).
        Suporta separação de vias Long/Short e utiliza a estrutura de Tuplas.
        """
        # 1. Configurações de Separação (L/S)
        # Busca no sm_mm_map se o rebalanceamento deve tratar Long e Short como entidades separadas
        global_sm_mm = self.portfolio.sm_mm_map.get("managers", {})
        global_sep_ls = global_sm_mm.get("separate_ls", False)

        entities_to_allocate = []
        model_map = {} # Mapeia 'Entidade_String' -> (Tupla_Original, Via)

        # 2. Identificação das Entidades de Risco
        # sim_data agora é { (op, model): { 'both': {...}, 'long': {...} } }
        for m_name, m_info in hierarchy['models'].items():
            if not m_info.get('active', True):
                continue
            
            # Verifica se este modelo específico tem regra de separação L/S
            m_config = self.portfolio.sm_mm_map.get("models", {}).get(m_name, {})
            m_sep_ls = m_config.get("managers", {}).get("separate_ls", global_sep_ls)
            
            m_data = sim_data.get(m_name)
            if not m_data:
                continue

            # Se houver separação e os dados existirem, tratamos Long e Short como ativos distintos no HRP
            if m_sep_ls and "long" in m_data and "short" in m_data:
                # Criamos nomes únicos para as colunas da matriz
                l_ent, s_ent = f"{m_name}_long", f"{m_name}_short"
                
                entities_to_allocate.extend([l_ent, s_ent])
                model_map[l_ent] = (m_name, "long")
                model_map[s_ent] = (m_name, "short")
            else:
                # Caso contrário, usamos o retorno agregado (both)
                ent_name = f"{m_name}_both"
                entities_to_allocate.append(ent_name)
                model_map[ent_name] = (m_name, "both")

        # Fallback: Se não houver entidades suficientes para correlacionar
        if len(entities_to_allocate) < 2:
            for ent in entities_to_allocate:
                m_n, _ = model_map[ent]
                hierarchy['models'][m_n]['weight'] = 1.0
            return hierarchy

        # 3. Construção da Matriz de Retornos
        lookback = self.reb_lookback # Geralmente 63 ou 126
        start_idx = max(0, i - lookback)
        matrix_data = {}

        for ent in entities_to_allocate:
            m_n, side = model_map[ent]
            # Acessa os retornos via tupla e via (long/short/both)
            # sim_data[m_n][side]['data'] é um numpy array
            series = sim_data[m_n][side]['data']
            
            # Pegamos o slice temporal e transformamos em 1D
            retornos = series[start_idx : i].flatten()
            
            # Proteção contra séries incompletas ou constantes
            if len(retornos) < (lookback * 0.7) or np.all(retornos == 0):
                # Se um modelo não tem dados, ele não entra no HRP para não distorcer a matriz
                continue
                
            matrix_data[ent] = retornos

        # 4. Cálculo do HRP
        if len(matrix_data) < 2:
            return hierarchy # Mantém pesos atuais se não houver dados para correlação

        try:
            import polars as pl
            from indicators.HRP import HRP # Certifique-se de que o caminho está correto

            df_returns = pl.DataFrame(matrix_data).fill_null(0.0)
            
            hrp_engine = HRP()
            # O método calculate_weights deve aceitar um DataFrame ou Matrix
            weights_dict = hrp_engine.calculate_weights_from_matrix(df_returns)

        except Exception as e:
            print(f"⚠️ Erro no cálculo HRP no índice {i}: {e}")
            return hierarchy

        # 5. Distribuição de Pesos para a Hierarchy
        # Resetamos os pesos dos modelos ativos para garantir que a soma seja limpa
        for m_name in hierarchy['models']:
            if hierarchy['models'][m_name].get('active', True):
                hierarchy['models'][m_name]['weight'] = 0.0
                if 'side_weights' in hierarchy['models'][m_name]:
                    hierarchy['models'][m_name]['side_weights'] = {}

        # Atribuição dos novos pesos calculados
        for ent, weight in weights_dict.items():
            m_n, side = model_map[ent]
            
            if side == "both":
                hierarchy['models'][m_n]['weight'] = weight
            else:
                # Se for separado, guardamos o peso individual da via e somamos no total do modelo
                hierarchy['models'][m_n].setdefault('side_weights', {})[side] = weight
                hierarchy['models'][m_n]['weight'] += weight

        return hierarchy

    def _default_main(self, i, step_dt, hierarchy: dict, indicator_pool: dict, port_returns: dict, key) -> bool:
        
        # Default uses aggr of models for Portfolio Level
        sim_data = self.get_data(key=key, lookback=self.reb_lookback, data_type="aggr", side="both") 

        hierarchy = self.rank(i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)
        hierarchy = self.filter(i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)
        print(hierarchy)
        hierarchy = self.rebalance(i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)
        print(hierarchy)
        return hierarchy
   
#||=========================================================================================||

    """ Dt execution framework

    1. Check current tradable Models
    -> REBALANCE

    2. New Rank generated with updated data
    3. Needs to remove any Models? if yes then close or keep positions open by SM rules? != MM rules
    4. Needs to add any Models? 
    

    """


















