"""
# Money Management Algorithm (SMM / MMM / PMM) - Base class for all Money Management
Função: controlar risco, exposição e alocação de capital.
Camadas:
SMM (Strategy Money Management): define quanto alocar por trade dentro da estratégia.
MMM (Model Money Management): define quanto cada estratégia do modelo recebe.
PMM (Portfolio Money Management): define quanto cada modelo recebe do portfólio.
"""

import polars as pl, uuid, math
from typing import Literal, Dict, Optional, Callable, List
from dataclasses import dataclass, field
from Indicator import Indicator
from BaseClass import BaseClass, BaseManager

@dataclass
class MoneyManagerParams:
    name: str = field(default_factory=lambda: f'mm_{uuid.uuid4()}')

    capital: float=100000.0
    max_capital_exposure: float=1.0
    
    reb_frequency: Literal["tick", "daily", "weekly", "monthly", "yearly", "never"] = "weekly"
    reb_lookback: int=252 # If len < lookback then [:idx]
    reb_lookback_period_type: Literal["tick", "day", "week", "month", "year"]="day" # 252 what? ticks, days?
 
    # Dados externos para MM (Ex: volatilidade do mercado, regime de juros)
    # Agora usa Polars DataFrame
    assets: Dict[str, pl.DataFrame] = field(default_factory=dict)

    # Customizable parameters for specific System Managers (Ex: thresholds para desativar modelos, regras de ativação, etc)
    params: Dict = field(default_factory=dict) 
    
    # Indicadores específicos para balanceamento de ativos/modelos
    indicators: Optional[Dict[str, Indicator]] = field(default_factory=dict) 

    # Plugin functions for custom model hierarchy rules and rebalancing logic
    fn_pre_compute:     Optional[Callable] = None   # (history: Dict[str, pl.DataFrame]) -> None
    fn_allocate:        Optional[Callable] = None   # (context: dict) -> Dict[str, float]
    fn_size:            Optional[Callable] = None   # (context: dict) -> List[str]
    fn_risk_guard:      Optional[Callable] = None   # (context: dict) -> List[str]
    fn_main:            Optional[Callable] = None   # (model_name: str, context: dict) -> bool

class MoneyManager(BaseClass, BaseManager): # Classe base para SMM, MMM e PMM
    def __init__(self, mm_params: MoneyManagerParams):
        super().__init__()
        self.name = mm_params.name

        self.capital = mm_params.capital

        self.cash = self.capital
        self.max_capital_exposure = mm_params.max_capital_exposure
        self.allocated_margin = 0.0

        self.total_equity = self.cash + self.allocated_margin + 0.0
        self.available_margin = self.total_equity * self.max_capital_exposure - self.allocated_margin

        self.reb_frequency = mm_params.reb_frequency
        self.reb_lookback = mm_params.reb_lookback
        self.reb_lookback_period_type = mm_params.reb_lookback_period_type
        
        # Custom Rules & Data
        self.assets = mm_params.assets
        self.params = mm_params.params
        self.indicators = mm_params.indicators

        # Funções plugáveis — usa custom se passado, senão usa default interno
        self._fn_pre_compute    = mm_params.fn_pre_compute
        self._fn_allocate       = mm_params.fn_allocate
        self._fn_size           = mm_params.fn_size
        self._fn_risk_guard     = mm_params.fn_risk_guard
        self._fn_main           = mm_params.fn_main

        self.portfolio = None

#||=========================================================================================||

    # def allocate(self, i, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key) -> Dict[str, float]:
    #     # Ranks each model by metric defined in model_hierarchy. Returns dict[model_name: score]
    #     return self._call(self._fn_allocate, self._default_allocate, i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)

    # def size(self, i, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key) -> List[str]:
    #     # Removes models that don't pass the filter function
    #     # Returns list of model_names that are active
    #     return self._call(self._fn_size, self._default_size, i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)

    # def risk_guard(self, i, step_dt, hierarchy: dict, indicator_pool: dict, sim_data: dict, port_returns: dict, key) -> List[str]:
    #     # Orchestrates rank -> filter -> selection
    #     # Returns ordered list of active models
    #     return self._call(self._fn_risk_guard, self._default_risk_guard, i, step_dt, hierarchy, indicator_pool, sim_data, port_returns, key)

    # def main(self, i, step_dt, key) -> bool:
    #     # Called every datetime for each model and asset
    #     # Returns True if model can operate now
    #     return self._call(self._fn_main, self._default_main, i, step_dt, self.portfolio.hierarchy, self.portfolio.indicator_pool, self.portfolio.portfolio_returns, key)

    def generate_target_allocation(self, 
                                   candidates: List[dict],
                                   current_positions: Dict[tuple, dict],
                                   total_equity: float,
                                   competition_mode: Literal["proportional", "top_down"]="top_down",
                                   liquidity_reserve_pct: float=0.0) -> List[dict]:
        # Receives pre-candidate's list from Portfolio, calculates prerequisites for each one and applies\
        #cutoff/reduction rules in case of lack of global margin

        if not candidates and not current_positions: return []

        # Applies liquidity reserve
        usable_capital = total_equity * (1 - liquidity_reserve_pct)

        # Calculates theoretical required capital for each pre-candidate
        total_requested = 0.0
        for can in candidates: # Weight is final result from (Portf * Model * Strat * Asset)
            can["req_capital"] = usable_capital * can["total_weight"]
            total_requested += can["req_capital"]
        approved_targets = []

        # Competition and Margin filter
        if total_requested <= usable_capital or total_requested == 0.0:
            approved_targets = candidates # Has margin for all pre-candidates 
        else: # Lacks margin for all, then filters
    
            # 1. Distributes usable_capital between all equally
            if competition_mode == "proportional":
                reduction_factor = usable_capital / total_requested
                for can in candidates:
                    can["req_capital"] *= reduction_factor 
                    approved_targets.append(can)

            # 2. Orders by score, survival of the fittest
            elif competition_mode == "top_down":
                candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
                available = usable_capital

                for can in candidates:
                    if available <= 0:
                        can["req_capital"] = 0.0
                        approved_targets.append(can)
                        continue

                    if can["req_capital"] <= available:
                        available -= can["req_capital"]
                    else:
                        can["req_capital"] = available
                        available = 0.0

                    approved_targets.append(can)

        # Order Generation
        final_orders = []
        candidate_keys = set()

        for can in approved_targets:
            c_key = can["c_key"]
            candidate_keys.add(c_key)
            
            req_cap = can["req_capital"]
            min_margin = can["margin_required"]
            min_lot = can["min_trade_lot_size"]
            lot_step = can.get("lot_step", min_lot)

            # Calculates lot target
            if req_cap > 0 and min_margin > 0:
                # Uses default leverage/lot_size logic
                raw_lot = self.calculate_lot_size(min_lot, min_margin, req_cap, lot_step)
                target_lot = raw_lot
            else:
                target_lot = 0.0
            
            # Current lot and margin that the system has retained for this asset
            current_pos = current_positions.get(c_key, {})
            current_lot = current_pos.get("lot_size", 0.0)

            # NOTE WIP below
            lot_multiplier = target_lot / min_lot if min_lot > 0 else 0.0
            actual_margin = lot_multiplier * min_margin
            delta_lot = target_lot - current_lot

            # Classifies order intention 
            if target_lot > 0 and current_lot == 0:
                order_type = "entry"
            elif target_lot == 0 and current_lot > 0:
                order_type = "exit"
            elif delta_lot > 0:
                order_type = "scale_in"
            elif delta_lot < 0:
                order_type = "scale_out"
            else:
                order_type = "hold"

            can["order_type"] = order_type
            can["target_lot"] = target_lot
            can["delta_lot"] = delta_lot
            can["allocated_margin"] = actual_margin

            final_orders.append(can)

        # Forced exits handling
        for c_key, pos_data in current_positions.items():
            if c_key not in candidate_keys:
                final_orders.append({
                    "c_key": c_key,
                    "target_weight": 0.0,
                    "target_lot": 0.0,
                    "delta_lot": -pos_data["lot_size"],
                    "allocated_margin": 0.0,
                    "order_type": "exit",
                    "event": "exit",
                    "trade_data": {"pnl": 0.0, "perc": 0.0} # Will be injected into real Portfolio value
                })

        return final_orders



    def update_states(self, active_positions):
        # Updates Equity and Margin available based on pnl
        floating_pnl = sum(pos.get("pnl", 0.0) for pos in active_positions.values())
        self.total_equity = self.cash + self.allocated_margin + floating_pnl
        self.available_margin = (self.total_equity * self.max_capital_exposure) - self.allocated_margin

    def calculate_position_size(self, global_weight):
        # Calculates nominal financial capital for candidate in $
        return self.total_equity * global_weight

    def calculate_lot_size(self, min_lot, min_margin_required, allocated_capital, lot_step):
        # Converts financial capital in operational lot size
        if min_margin_required <= 0:
            return min_lot
        
        # /leverage already done in cpp backtester for min_lot
        lot_mult = allocated_capital / min_margin_required
        ideal_lot_raw = lot_mult * min_lot

        ideal_lot = math.floor(ideal_lot_raw / lot_step) * lot_step

        return max(ideal_lot, min_lot) # Only >= minimum lot size

#||=========================================================================================||


    # def __repr__(self):
    #     return f"<{self.__class__.__name__} name={self.name} capital={self.capital}>"