from dataclasses import dataclass, field
from SystemManager import SystemManager, SystemManagerParams
from typing import List, Optional, Dict, Literal, Callable, Union
#from Backend.core import Asset
import polars as pl

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
    reb_frequency: Literal["daily", "weekly", "monthly", "quarterly", "yearly","never"] = "weekly"
    reb_lookback_n: int = 1
    reb_deviation_func: Optional[Dict[str, Callable]] = None # Function that defines the deviation threshold needed for rebalancing (e.g., 5% deviation from target allocation)
    reb_close_open_trades_on_rebalance: bool = False

    # Plugin functions for custom model hierarchy rules and rebalancing logic
    fn_pre_compute:     Optional[Callable] = None   # (history: Dict[str, pl.DataFrame]) -> None
    fn_rank:            Optional[Callable] = None   # (context: dict) -> Dict[str, float]
    fn_filter:          Optional[Callable] = None   # (context: dict) -> List[str]
    fn_rebalance:       Optional[Callable] = None   # (context: dict) -> List[str]
    fn_should_execute:  Optional[Callable] = None   # (model_name: str, context: dict) -> bool

class PortfolioSystemManager(SystemManager): # Manages portfolio's model hierarchy 
    def __init__(self, pm_params: PortfolioSystemManagerParams):
        super().__init__(pm_params)

        self.reb_metric                         = pm_params.reb_metric
        self.model_hierarchy                    = dict(pm_params.model_hierarchy)
        self.max_active_models                  = pm_params.max_active_models
        self.reb_method                         = pm_params.reb_method
        self.reb_frequency                      = pm_params.reb_frequency
        self.reb_lookback_n                     = pm_params.reb_lookback_n
        self.reb_close_open_trades_on_rebalance = pm_params.reb_close_open_trades_on_rebalance

        # Funções plugáveis — usa custom se passado, senão usa default interno
        self._fn_pre_compute    = pm_params.fn_pre_compute
        self._fn_rank           = pm_params.fn_rank
        self._fn_filter         = pm_params.fn_filter
        self._fn_rebalance      = pm_params.fn_rebalance
        self._fn_should_execute = pm_params.fn_should_execute

        self._pre_cache: Dict = {}   # cache de indicadores pre-calculados
        self._active_models: List[str] = []

    # ── Helper central ───────────────────────────────────────────────────────

    def _call(self, custom_fn: Optional[Callable], default_fn: Callable, *args, **kwargs):
        # Calls custom_fn if exists, else default
        return custom_fn(*args, **kwargs) if custom_fn else default_fn(*args, **kwargs)
    
    # ── Pre-Simulation ───────────────────────────────────────────────────────

    def pre_compute(self, history: Dict[str, pl.DataFrame]) -> None:
        # Runs only unce at the beginning of the simulation, can be used to pre-calculate indicators or do any setup needed before the simulation starts. 
        self._call(self._fn_pre_compute, self._default_pre_compute, history)

    def _default_pre_compute(self, context: Dict[str, pl.DataFrame]) -> None:

        # NEEDS ADAPTATION TO PS_PARAMR

        # Calculates sm_indicators defined as when="pre" 
        assets = context.get("assets", {})
        for ind_name, ind_obj in (self.sm_indicators or {}).items():
            if getattr(ind_obj,"when", "pre") == "pre":
                if ind_obj.asset is not None: # Computes ind for specific asset only
                    asset_class = assets.get(ind_obj.asset)
                    asset_df = asset_class.data_get(ind_obj.timeframe) if asset_class else None
                    if asset_class is not None and asset_df is not None:
                        self._pre_cache[ind_name] = ind_obj.calculate(asset_df)
                else: # Computes ind for all assets 
                   for asset_name in self.sm_assets.items():
                        asset_class = assets.get(asset_name) # history has assets class
                        asset_df = asset_class.data_get(ind_obj.timeframe) if asset_class else None
                        if asset_class is not None and asset_df is not None:
                                self._pre_cache[ind_name] = ind_obj.calculate(asset_df)

    # ── Every reb_frequency ───────────────────────────────────────────

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

    # Default implementations (used if no custom function is passed in the params)

    def _default_rank(self, context: dict) -> Dict[str, float]:
        history  = context.get("history", {})
        lookback = self.reb_lookback_n
        scores   = {}

        for model, df in history.items():
            if df.is_empty():
                scores[model] = -float("inf")
                continue
            tail = df.tail(lookback)["pnl"]
            if self.reb_metric == "sharpe":
                scores[model] = float(tail.mean() / (tail.std() + 1e-9))
            elif self.reb_metric == "pnl_dd":
                eq  = tail.cum_sum()
                dd  = float((eq.cum_max() - eq).max() + 1e-9)
                scores[model] = float(tail.sum()) / dd
            else:  # pnl
                scores[model] = float(tail.sum())

        return scores

    def _default_filter(self, context: dict) -> List[str]:
        models = context.get("models", []) 
        return models # By default doesn't filter out any model

    def _default_rebalance(self, context: dict) -> List[str]:
        scores = self.rank(context)
        context["scores"] = scores

        passed = self.filter(context)
        descending = self.model_hierarchy.get("order_by", "highest") == "highest"
        ranked = sorted(passed, key=lambda m: scores.get(m, -float("inf")), reverse=descending)

        if self.max_active_models:
            ranked = ranked[:self.max_active_models]

        self._active_models = ranked
        return ranked

    # ── Every Datetime [i] ───────────────────────────────────────────────

    def should_execute(self, model_name: str, context: dict) -> bool:
        # Called every datetime for each model and asset
        # Returns True if model can operate now
        return self._call(self._fn_should_execute, self._default_should_execute, model_name, context)
    
    def _default_should_execute(self, model_name: str, context: dict) -> bool:

        # WIP live indicators implementation

        for name, ind in (self.sm_indicators or {}).items():
            if getattr(ind, "when", "pre") == "live":
                result = ind.calculate(context.get("live_data"))
                if not result: 
                    return False
        return True



    """ Dt execution framework

    1. Check current tradable Models
    -> REBALANCE

    2. New Rank generated with updated data
    3. Needs to remove any Models? if yes then close or keep positions open by SM rules? != MM rules
    4. Needs to add any Models? 
    

    """


















