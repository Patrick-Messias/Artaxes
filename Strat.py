from typing import Dict, Optional, List, Callable, Literal
from dataclasses import dataclass, field
import uuid

from StratMoneyManager import StratMoneyManager
from Indicator import Indicator
from Walkforward import Walkforward


# ═════════════════════════════════════════════════════════════════════════════
# EXECUTION SETTINGS
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ExecutionSettings:
    # ── Posições ──────────────────────────────────────────────────────────────
    hedge:                      bool            = False
    strat_num_pos:              List[int]       = field(default_factory=lambda: [1, 1])   # [long, short]
    strat_max_num_pos_per_day:  List[int]       = field(default_factory=lambda: [-1, -1]) # -1 = sem limite

    # ── Ordem ────────────────────────────────────────────────────────────────
    order_type:                         str  = 'market'   # 'market' | 'limit' | 'stop'
    limit_order_base_calc_ref_price:    str  = 'open'
    slippage:                           float = 0.0       # em ticks — multiplicado pelo tick do Asset em _operation
    commission:                         float = 0.0       # em ticks — multiplicado pelo tick do Asset em _operation

    # ── Backtest mode ─────────────────────────────────────────────────────────
    backtest_mode: Literal['ohlc', 'close-close', 'open-open', 'avg_price'] = 'ohlc'
    # 'ohlc'        → padrão, verifica SL/TP intrabar via high/low
    # 'close-close' → executa apenas no fechamento, sem verificação intrabar
    # 'open-open'   → executa apenas na abertura
    # 'avg_price'   → preço = (o+h+l+c)/4

    # ── Day Trade / Horários ──────────────────────────────────────────────────
    day_trade:                      bool                = False
    timeTI:                         Optional[str]       = None   # "HH:MM" — início entrada
    timeEF:                         Optional[str]       = None   # "HH:MM" — fim entrada
    timeTF:                         Optional[str]       = None   # "HH:MM" — força fechamento
    next_index_day_close:           bool                = False

    # ── Filtros de calendário ─────────────────────────────────────────────────
    day_of_week_close_and_stop_trade:   Optional[List[int]] = None  # [0=seg .. 6=dom]
    timeExcludeHours:                   Optional[List[int]] = None
    dateExcludeTradingDays:             Optional[List[int]] = None
    dateExcludeMonths:                  Optional[List[int]] = None

    # ── Preenchimento / resolução ─────────────────────────────────────────────
    fill_method:            str     = 'ffill'
    fillna:                 object  = 0
    trade_pnl_resolution:   str     = 'daily'   # 'daily' | 'trade'

    # ── Debug ─────────────────────────────────────────────────────────────────
    print_logs: bool = False


# ═════════════════════════════════════════════════════════════════════════════
# STRAT PARAMS
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class StratParams:
    name:               str                                             = field(default_factory=lambda: f'strat_{uuid.uuid4()}')
    operation:          Optional[Walkforward] = None
    params:             Dict                                            = field(default_factory=dict)
    execution_settings: ExecutionSettings                               = field(default_factory=ExecutionSettings)
    strat_money_manager: Optional[StratMoneyManager]                   = None
    indicators:         Dict[str, Indicator]                           = field(default_factory=dict)
    signals:            Optional[Callable]                             = None


# ═════════════════════════════════════════════════════════════════════════════
# STRAT
# ═════════════════════════════════════════════════════════════════════════════

class Strat:
    """
    Encapsula uma estratégia de trading — sinais, indicadores, execução e MM.

    operation:          Walkforward | Optimization | None
    params:             dict de parâmetros (fixos ou ranges para combinações)
    execution_settings: regras de execução (ordem, horário, day trade, etc.)
    strat_money_manager: sizing de lote — None usa neutro (lot=1.0)
    indicators:         dict {nome: Indicator} — calculados antes dos sinais
    signals:            função (df, params) → dict de sinais para o C++
    """

    def __init__(self, strat_params: StratParams):
        self.name                = strat_params.name
        self.operation           = strat_params.operation
        self.params              = strat_params.params
        self.execution_settings  = strat_params.execution_settings
        self.strat_money_manager = strat_params.strat_money_manager
        self.indicators          = strat_params.indicators
        self.signals             = strat_params.signals

    def __repr__(self) -> str:
        op   = type(self.operation).__name__ if self.operation else 'None'
        smm  = type(self.strat_money_manager).__name__ if self.strat_money_manager else 'None'
        inds = list(self.indicators.keys())
        return (f"<Strat '{self.name}' | op={op} | "
                f"indicators={inds} | SMM={smm} | "
                f"mode={self.execution_settings.backtest_mode}>")