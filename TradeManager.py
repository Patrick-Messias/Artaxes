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
import BaseClass, Indicator, uuid

@dataclass
class Trade_Management_Parameters():
    name: str=f'tm_{str(uuid.uuid4())}'
    trades: dict[Trade]={}

class Trade_Management(BaseClass): 
    def __init__(self, tm_params: Trade_Management_Parameters):
        super().__init__()
        self.name = tm_params.name
        self.trades = tm_params.trades

    def _get_trade_id(self, trade: Trade): # Returns trade's id (str)
        if isinstance(trade, str): 
            return trade

        if hasattr(trade, "id"): 
            return trade.id
        else: 
            raise ValueError("Invalid trade - Has to be Trade object with 'id' attribute")

    def add_trade(self, trade: Trade): # Adds single trade to trades dict
        if not hasattr(trade, "id"):
            trade.id = str(uuid.uuid4())

        trade_id = trade.id

        if trade_id in self.trades:
            raise ValueError(f"Trade with id {trade_id} already in trades dict")

        return self.trades[trade_id] = trade

    def remove_trade(self, trade: Trade): # Removes single trade from trades dict
        trade_id = self._get_trade_id(trade)

        if trade_id in self.trades:
            del self.trades[trade_id]
            return True
        else:
            return False

    def get_trades(self, atribute=None): # Gets open and closed trades from trades dict, atribute=id
        if atribute is None:
            return list(self.trades.values())

        if isinstance(atribute, str) and atribute in self.trades:
            return [self.trades[atribute]]

        result = []
        for trade in self.trades.values():
            if hasattr(trade, "atribute") and trade.atribute == atribute:
                result.append(trade)

        return result

    def get_open_trades(self):
        return [t for t in self.trades.values() if getattr(t, "status", None) == "open"]

    def get_closed_trades(self):
        return [t for t in self.trades.values() if getattr(t, "status", None) == "closed"]

    def get_trades_by_asset(self, asset: Asset):
        return [t for t in self.trades.values() if getattr(t, "asset", None) == asset]



