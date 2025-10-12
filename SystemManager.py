"""
# System Management Algorithm (SMA) - Base class for all System Management
Função: orquestrar o comportamento do sistema em tempo de execução.
Liga/desliga Strats ou Models conforme regras globais.
Define quais combinações (Model + Asset + Strat) estão ativas.
Pode implementar lógica de auto-adaptação (ex: desativar modelos com drawdown alto).
Atua sobre os níveis superiores (controla quem “fala” com o PMM e o TM).
🔹 Pense nele como o “cérebro administrativo” do sistema.
"""

from typing import Dict, Optional, Callable
from dataclasses import dataclass, field
import BaseClass, Indicator, uuid

@dataclass
class SystemManagerParameters():
    name: str = field(default_factory=lambda: f'sm_{uuid.uuid4()}')
    
    sm_external_data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    sm_indicators: Optional[Dict[str, 'Indicator']] = field(default_factory=dict)
    sm_rules: Optional[Dict[str, Callable]] = field(default_factory=dict)

class SystemManager(BaseClass): 
    def __init__(self, system_params: SystemManagerParameters):
        self.name = system_params.name
        
        # Custom Rules
        self.sm_external_data = dict(system_params.sm_external_data) # For external data like CDT not present during Strat or Model construction
        self.sm_indicators = dict(system_params.sm_indicators)
        self.sm_rules = dict(system_params.sm_rules)