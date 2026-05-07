from dataclasses import dataclass, field
#from Backend.core import Asset
from MoneyManager import MoneyManager, MoneyManagerParams
from typing import Optional, Dict, Literal, Callable, List
import polars as pl, numpy as np

@dataclass
class PortfolioMoneyManagerParams(MoneyManagerParams):
    # Allocation
    alo_allocation: Optional[Dict[str, float]]=None # Ex: {"Model_A": 0.5, "Model_B": 0.3, "Model_C": 0.2} -> 50% do capital para Model_A, 30% para Model_B e 20% para Model_C

    # Rebalancing
    reb_metric: Literal["pnl", "pnl_dd", "sharpe"] = "pnl" # Metric used for performance-based rebalancing (if reb_method == "performance")
    reb_method: Literal["fixed", "equal_weight", "risk_parity", "performance"] = "fixed"
    reb_deviation_func: Optional[Dict[str, Callable]] = None # Function that defines the deviation threshold needed for rebalancing (e.g., 5% deviation from target allocation)

class PortfolioMoneyManager(MoneyManager): # Manages Model's risk and money management
    def __init__(self, pmm_params): # PMM(Portfolio) > MMM(Model) > MMA(Strat)
        super().__init__(pmm_params)
        self.alo_allocation = pmm_params.alo_allocation

        self.reb_metric = getattr(pmm_params, 'reb_metric', 'pnl')
        self.reb_method = getattr(pmm_params, 'reb_method', 'fixed')
        self.reb_deviation_func = getattr(pmm_params, 'reb_deviation_func', None)

        # Travas de risco e alavancagem padrão (caso não venham no pmm_params)
        self.max_portfolio_leverage = getattr(pmm_params, 'max_portfolio_leverage', 1.0)
        self.max_drawdown_limit = getattr(pmm_params, 'max_drawdown_limit', -0.2)

        self._pre_cache: Dict = {}

#||=========================================================================================||

    def _default_pre_compute(self, global_assets, timeline, aggr_ret, indicator_pool, param_sets, manager_level_key) -> dict:
        # By Default doesn't calculate anything else, but can be used to prepare signals or other stuff != indicators
        return indicator_pool
          
    # ── Every Datetime [i] ───────────────────────────────────────────────

    def _default_allocate(self, i, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key):
        # If fixed, PMM overrides PSM
        if self.reb_method == "fixed" and self.alo_allocation:
            for m_key, m_info in self.portfolio.iter_hierarchy(target_level="models"):
                if m_info.get('active', True):
                    name = f"{m_key[0]}_{m_key[1]}"
                    weight = self.alo_allocation.get(name, 0.0)
                    for s in ["both", "long", "short"]:
                        if s in m_info: m_info[s]['weight'] = weight
        return hierarchy

    def _default_size(self, i, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key):
        # Aplica a alavancagem global permitida pelo Portfolio
        mult = self.max_portfolio_leverage
        
        for m_key, m_info in self.portfolio.iter_hierarchy(target_level="models"):
            if m_info.get('active', True):
                for s in ["both", "long", "short"]:
                    if s in m_info:
                        m_info[s]['weight'] *= mult

        return hierarchy

    def _default_risk_guard(self, i, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key):
        # Verifica Drawdown máximo do Portfolio após 20 observações (trades/períodos)
        if key in port_returns:
            rets = port_returns[key]
            
            if len(rets) > 20:
                # Calcula a curva de capital acumulada e o drawdown
                cum_rets = np.cumsum(rets)
                peak = np.maximum.accumulate(cum_rets)
                dd = cum_rets - peak
                current_dd = dd[-1]

                # Se o DD passar do limite (ex: -0.2 para -20%), trava tudo
                if current_dd <= self.max_drawdown_limit:
                    for m_key, m_info in self.portfolio.iter_hierarchy(target_level="models"):
                        m_info['active'] = False
                        # Zera os pesos por precaução
                        for s in ["both", "long", "short"]:
                            if s in m_info:
                                m_info[s]['weight'] = 0.0
                    
                    # Print opcional para debug no terminal
                    print(f"!!! [PMM RISK GUARD] Kill-Switch acionado em {step_dt} | DD Atual: {current_dd:.2%} | Limite: {self.max_drawdown_limit:.2%}")
        
        return hierarchy

    def _default_main(self, i, step_dt, hierarchy: dict, indicator_pool: dict, port_returns: dict, key) -> bool:
        # Default uses aggr of models for Portfolio Level
        lookback = getattr(self, 'reb_lookback', 63)
        sim_data = self.get_data(key=key, lookback=lookback, data_type="aggr", side="both")

        hierarchy = self.allocate(i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)
        hierarchy = self.size(i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)
        hierarchy = self.risk_guard(i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)

        return hierarchy

#||=========================================================================================||










