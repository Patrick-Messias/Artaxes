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
import BaseClass, Indicator

@dataclass
class System_Management_Algorithm_Parameters():
    name: str='unnamed_pma'
    
    model_hierarchy: str='default'
    rebalance_frequency: str='weekly'

    external_data: Dict[str, pd.DataFrame] = field(default_factory=dict)=None

    indicators: Optional[Dict[str, Indicator]] = field(default_factory=dict) # For Model/Asset Balancing
    pma_rules: Optional[Dict[str, Callable]] = field(default_factory=dict)

class System_Management_Algorithm(BaseClass): 
    def __init__(self, pma_params: System_Management_Algorithm_Parameters):
        super().__init__()
        self.name = pma_params.name
        
        self.model_hierarchy = pma_params.model_hierarchy
        self.rebalance_frequency = pma_params.rebalance_frequency

        self.external_data = pma_params.external_data # For external data like CDT

        # Custom Rules
        self.indicators = pma_params.indicators
        self.pma_rules = pma_params.pma_rules