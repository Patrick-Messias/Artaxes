from dataclasses import dataclass, field
from SystemManager import SystemManager, SystemManagerParams
from typing import Optional, Callable, Dict, List

@dataclass
class StratSystemManagerParams(SystemManagerParams):
    strat_hierarchy: dict = field(default_factory=lambda: {"order_by": 'highest', "metric": 'profit_perc'})
    rebalance_frequency: str = 'weekly'
    strat_lookback_n: int = 1
    close_open_trades_on_rebalance: bool = False

class StratSystemManager(SystemManager): 
    def __init__(self, ssm_params: SystemManagerParams):
        super().__init__(ssm_params) 
        
        self.strat_hierarchy = dict(ssm_params.strat_hierarchy)
        self.rebalance_frequency = ssm_params.rebalance_frequency
        self.strat_lookback_n = ssm_params.strat_lookback_n
        self.close_open_trades_on_rebalance = ssm_params.close_open_trades_on_rebalance

#||=========================================================================================||

    def _default_pre_compute(self, global_assets, timeline, sim_data, aggr_ret, indicator_pool, param_sets) -> dict:

        # By Default doesn't calculate anything else, but can be used to prepare signals or other stuff != indicators
        
        return indicator_pool, sim_data
                       
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

    def main(self, step_dt, hierarchy: dict, op_data: dict, port_returns: dict) -> bool:
        # Called every datetime for each model and asset
        # Returns True if model can operate now
        return self._call(self._fn_main, self._default_main, step_dt, hierarchy, op_data, port_returns)
    
    def _default_main(self, step_dt, hierarchy: dict, op_data: dict, port_returns: dict) -> bool:

        # Calculates Live Indicators

        # Rebalances

        return hierarchy

#||=========================================================================================||


















