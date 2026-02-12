import numpy as np, itertools, importlib
from typing import Dict, Optional, Union, List, Callable
from dataclasses import dataclass, field
from BaseClass import BaseClass
from Asset import Asset, Asset_Portfolio
from finta import TA
import uuid, inspect

from MoneyManager import MoneyManager, MoneyManagerParams
from StratMoneyManager import StratMoneyManager, StratMoneyManagerParams
from Indicator import Indicator

from Backtest import Backtest, BacktestParams
from Optimization import Optimization
from Walkforward import Walkforward

# Import will be added later to avoid circular import
# from StratMoneyManager import StratMoneyManager

@dataclass
class ExecutionSettings:
    hedge: bool=False
    strat_num_pos: list[int]=field(default_factory=lambda: [1,1])
    order_type: str='market'
    offset: float=0.0

@dataclass
class TimeSettings:
    day_trade: bool = False
    timeTI: Optional[list[int]] = None
    timeEF: Optional[list[int]] = None
    timeTF: Union[bool, List[int]] = False
    next_index_day_close: bool = False
    day_of_week_close_and_stop_trade: Optional[List[int]] = None
    timeExcludeHours: Optional[List[int]] = None
    dateExcludeTradingDays: Optional[List[int]] = None
    dateExcludeMonths: Optional[List[int]] = None

""" 
@dataclass # # TradeManagementRules Eliminated, it must check in Strat's generate_signals() 
class TradeManagementRules:
    LONG: bool = True
    SHORT: bool = True
    TF: bool = True
    SL: bool = True
    TP: bool = False
    BE_pos: bool = False
    BE_neg: bool = False
    NB: int = -1
"""

@dataclass
class DataSettings: # Commenting dataframe to avoid repetition, additional_timeframes can be replaced by a func
    fill_method: str='ffill'
    fillna: object=0

@dataclass
class StratParams():
    name: str = field(default_factory=lambda: f'strat_{uuid.uuid4()}')
    operation: Union[Backtest, Optimization, Walkforward]=None
    #strat_support_assets: Optional[Dict[str, Asset]] = field(default_factory=None) #Dict[str, Dict[str, Union[str, List[str]]]] = field(default_factory=dict)

    params: Dict = field(default_factory=dict) 
    execution_settings: ExecutionSettings = field(default_factory=ExecutionSettings)
    data_settings: DataSettings = field(default_factory=DataSettings)
    mma_settings: MoneyManagerParams = field(default_factory=MoneyManagerParams) # If mma_rules=None then will use default or PMA or other saved MMA define in Operation. Else it creates a temporary MMA with mma_settings
    time_settings: TimeSettings = field(default_factory=TimeSettings)
    indicators: Dict[str, Indicator] = field(default_factory=dict) 

    signal_rules: Dict = field(default_factory=lambda: {
        'entry_long': None,
        'entry_short': None,
        'exit_tf_long': None,
        'exit_tf_short': None,
        'entry_long_limit_price': None,
        'entry_short_limit_price': None,
        'exit_sl_long_price': None,
        'exit_sl_short_price': None,
        'exit_tp_long_price': None,
        'exit_tp_short_price': None,
        'be_pos_long': None,
        'be_pos_short': None,
        'be_neg_long': None,
        'be_neg_short': None
    })

    strat_money_manager: Optional['StratMoneyManager'] = None

def call_rule_function(func, **kwargs):
    """Chama uma função com apenas os argumentos que ela realmente usa."""
    sig = inspect.signature(func)
    valid_args = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(**valid_args)
    
class Strat(BaseClass):
    def __init__(self, strat_params: StratParams):
        super().__init__()
        self.name = strat_params.name
        self.operation = strat_params.operation

        self.params = strat_params.params
        self.execution_settings = strat_params.execution_settings
        self.data_settings = strat_params.data_settings
        self.mma_settings = strat_params.mma_settings # If mma_rules=None then will use default or PMA or othe MMA define in Operation
        self.time_settings = strat_params.time_settings
        self.indicators = strat_params.indicators

        self.signal_rules = strat_params.signal_rules

        # StratMoneyManager is optional - if None, will use default or PMA/MMM from Operation
        self.strat_money_manager = strat_params.strat_money_manager

        self.data = None

    def __repr__(self):
        return self.__str__()





