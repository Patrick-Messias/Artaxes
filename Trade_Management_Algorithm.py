"""
# Trade Management Algorithm (TMA) - Centralizes all trades for an Operation
Função: supervisionar e registrar todas as operações abertas/fechadas.
Mantém histórico de trades, ordens, posições.
Faz netting, agregação e realização de PnL.
Centralizado: único módulo gerencia tudo, do Strat isolado ao Portfolio completo.
Pode ter subvisões filtradas (ex: trades de um Strat, trades de um Model, etc.).
🔹 Importante: TM é único e transversal — serve como “livro razão” de toda a atividade do sistema.
"""

from typing import Dict, Optional, Callable
from dataclasses import dataclass, field
import BaseClass, Indicator

@dataclass
class Trade_Management_Algorithm_Parameters():
    name: str='unnamed_pma'

class Trade_Management_Algorithm(BaseClass): 
    def __init__(self, tma_params: Trade_Management_Algorithm_Parameters):
        super().__init__()
        self.name = tma_params.name
