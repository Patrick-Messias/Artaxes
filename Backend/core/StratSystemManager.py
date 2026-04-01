from dataclasses import dataclass, field
from SystemManager import SystemManager, SystemManagerParams
from typing import Optional, Callable

@dataclass
class StratSystemManagerParams(SystemManagerParams):
    strat_hierarchy: dict = field(default_factory=lambda: {"order_by": 'highest', "metric": 'profit_perc'})
    rebalance_frequency: str = 'weekly'
    strat_lookback_n: int = 1
    close_open_trades_on_rebalance: bool = False

    # Plugin functions for custom model hierarchy rules and rebalancing logic
    fn_pre_compute:     Optional[Callable] = None   # (history: Dict[str, pl.DataFrame]) -> None
    fn_rank:            Optional[Callable] = None   # (context: dict) -> Dict[str, float]
    fn_filter:          Optional[Callable] = None   # (context: dict) -> List[str]
    fn_rebalance:       Optional[Callable] = None   # (context: dict) -> List[str]
    fn_should_execute:  Optional[Callable] = None   # (model_name: str, context: dict) -> bool

class StratSystemManager(SystemManager): 
    def __init__(self, sm_params: SystemManagerParams):
        super().__init__(sm_params) 
        
        self.strat_hierarchy = dict(sm_params.strat_hierarchy)
        self.rebalance_frequency = sm_params.rebalance_frequency
        self.strat_lookback_n = sm_params.strat_lookback_n
        self.close_open_trades_on_rebalance = sm_params.close_open_trades_on_rebalance

        self._fn_pre_compute    = sm_params.fn_pre_compute
        self._fn_rank           = sm_params.fn_rank
        self._fn_filter         = sm_params.fn_filter
        self._fn_rebalance      = sm_params.fn_rebalance
        self._fn_should_execute = sm_params.fn_should_execute


    def balance(self, strat_data): 
        return self._call(self._fn_rebalance, self._default_rebalance, strat_data)

    def _default_rebalance(self, strat_data): # Updates the hierarchy of the strat enside the model

        # Checks pma_rules on how to rebalance

        # Analyses current hierarchy and models performance based on rebalancing rules

        # Updates hierachy

        return None




















