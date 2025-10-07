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


    def close(self, close_params: dict):
        self.exit_price = close_params.get('exit_price', 0.0)
        self.exit_time = close_params.get('exit_time', '00:00:00')
        self.exit_reason = close_params.get('exit_reason', 'unknown')
        
        price_diff = (self.exit_price - self.entry_price) if self.direction == 'long' else (self.entry_price - self.exit_price)
        self.profit = price_diff * self.lot_size
        
        if self.stop_loss:
            risk = abs(self.entry_price - self.stop_loss)
            self.profit_r = price_diff / risk if risk != 0 else 0

    def open(self, open_params: dict):
        self.entry_price = open_params.get('entry_price', 0.0)
        self.entry_time = open_params.get('entry_time', '00:00:00')
        

    def get_trades_returns(self, trades: dict[Trade], type: str='perc'):
        trade_returns=[]
        
        for trade in trades:
            trade_returns.append(trade.)

        return trade_returns

