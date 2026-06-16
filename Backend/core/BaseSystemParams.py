# Holds defaults parameters for each Level of System Manager

import uuid
from typing import Literal, Set, Dict, Optional, Callable
from dataclasses import dataclass, field
from Indicator import Indicator

@dataclass
class BaseSystemParams(): 
    name: str = field(default_factory=lambda: f'sm_{uuid.uuid4()}')
    reb_frequency: Literal["tick", "daily", "weekly", "monthly", "yearly", "never"] = "weekly"
    reb_lookback: int=252
    reb_lookback_period_type: Literal["tick", "day", "week", "month", "year"]="day"

    # Custom Data, Rules for System Manager (Ex: Calendário Econômico, Sentimento, CDT)
    assets: Set[str] = field(default_factory=set)

    # Customizable parameters for specific System Managers (Ex: thresholds para desativar modelos, regras de ativação, etc)
    params: Dict = field(default_factory=dict) 

    # Indicadores administrativos (Ex: Medidores de Regime de Mercado)
    indicators: Optional[Dict[str, Indicator]] = field(default_factory=dict)

    # Funções plugáveis — usa custom se passado, senão usa default interno
    fn_pre_compute:     Optional[Callable] = None   # (history: Dict[str, pl.DataFrame]) -> None
    fn_rank:            Optional[Callable] = None   # (context: dict) -> Dict[str, float]
    fn_filter:          Optional[Callable] = None   # (context: dict) -> List[str]
    fn_rebalance:       Optional[Callable] = None   # (context: dict) -> List[str]
    fn_main:            Optional[Callable] = None   # (model_name: str, context: dict) -> bool

