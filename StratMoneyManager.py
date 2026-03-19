import numpy as np
import polars as pl
from dataclasses import dataclass, field
from typing import Optional, List, Literal
from Trade import Trade
from MoneyManager import MoneyManager, MoneyManagerParams


# ─────────────────────────────────────────────────────────────────────────────
# Params
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StratMoneyManagerParams(MoneyManagerParams):

    # ── Position/Lot Sizing ──────────────────────────────────────────────────────
    sizing_method: Literal["neutral", "fixed", "risk_per_trade", "pct_capital", "kelly", "var", "signal"] = "neutral"
    # "neutral"         → lot = 1.0 (padrão — Portfolio Sim sobrescreve se quiser)
    # "fixed"           → lot fixo em sizing_params["fixed_lot"]              — C++
    # "risk_per_trade"  → capital × risk_pct / (dist_ticks × tick_fin_val)    — C++
    # "pct_capital"     → (capital × pct) / (price × tick_fin_val)            — C++
    # "kelly"           → kelly fraction usando histórico acumulado no C++
    # "var"             → VaR usando histórico acumulado no C++
    # "signal"          → lot vem de custom_lot_size_long/short (pl.Series)   — Python → pool

    # ── Capital Management ───────────────────────────
    capital_method: Literal["fixed", "compound"] = "fixed"
    # "fixed"           → Always use initial capital, without compounding
    # "compound"        → Initial capital + (compound_fract=1.0 * profit)
    compound_fract: float = 1.0 # Used to adjust how much of profit is reinvested into new entries (!= lot_size)
    compound_fract_series: Optional[pl.Series] = None # if is not None then uses this

    sizing_params: dict = field(default_factory=lambda: {
        "fixed_lot":      1.0,    # para "fixed"
        "risk_pct":       0.01,   # para "risk_per_trade", "kelly" e "var"
        "risk_pct_min":   0.001,  # clamp mínimo (todos os métodos)
        "risk_pct_max":   0.05,   # clamp máximo (todos os métodos)
        "pct":            0.02,   # para "pct_capital"
        "kelly_weight":   0.25,   # fração kelly (ex: 0.25 = quarter-kelly)
        "var_confidence": 0.95,   # para "var"
        "min_trades":     30,     # mínimo de trades para kelly/var serem ativados no C++
    })

    # ── Referência de distância para risk_per_trade ───────────────────────────
    # Não é necessariamente o SL — pode ser ATR, tick, high[1], valor fixo, etc.
    # Prioridade: dist_signal_ref → dist_fixed → tick do asset (fallback no C++)
    dist_signal_ref: Optional[str] = None   # ex: "atr", "high[1]", "tick" — C++ resolve no fast_pool
    dist_fixed: Optional[float] = None      # distância fixa em pontos como fallback

    # ── Custom lot size via pl.Series (method="signal") ───────────────────────
    # Calculado em Python igual a um indicador, injetado no indicators_pool
    # C++ lê via fast_pool no open_trade — separado por lado
    custom_lot_size_long:  Optional[pl.Series] = None
    custom_lot_size_short: Optional[pl.Series] = None

# ─────────────────────────────────────────────────────────────────────────────
# StratMoneyManager
# ─────────────────────────────────────────────────────────────────────────────

class StratMoneyManager(MoneyManager):
    """
    Gerencia sizing de lote por Strat.

    Fluxo:
        Backtest individual:
            SMM definido     → calcula lot_size real → DailyResult reflete gestão da strat
            SMM não definido → lot = 1.0 → retorno neutro % para WF/Portfolio

        Walkforward Analysis:
            usa lot_size do DailyResult como está (benchmark da strat isolada)

        Portfolio Simulator:
            recebe capital alocado de MMM/PMM
            sobrescreve lot_size via calc_lot_size(capital=capital_alocado)
            mesmas regras do SMM, capital diferente

    Métodos C++ (backtest):
        neutral, fixed, risk_per_trade, pct_capital, kelly, var

    Métodos Python (pre-backtest):
        signal → custom_lot_size_long/short como pl.Series → indicators_pool

    Interfaces:
        to_sim_params()  → serializa para C++
        calc_lot_size()  → escalar para Portfolio Simulator (capital externo de MMM/PMM)
    """

    def __init__(self, params: StratMoneyManagerParams):
        super().__init__(params)

        self.sizing_method         = params.sizing_method
        self.sizing_params         = params.sizing_params

        self.capital_method         = params.capital_method
        self.compound_fract         = params.compound_fract
        self.compound_fract_series  = None

        self.dist_signal_ref       = params.dist_signal_ref
        self.dist_fixed            = params.dist_fixed
        self.custom_lot_size_long  = None
        self.custom_lot_size_short = None

    # ─────────────────────────────────────────────────────────────────────────
    # Interface C++ — serializa método e params para sim_params
    # ─────────────────────────────────────────────────────────────────────────
 
    def to_sim_params(self, capital: Optional[float] = None) -> dict:
        """
        Serializa o SMM para dict que vai em sim_params["money_manager"].
        C++ lê em money_manager.cpp e calcula lot_size no open_trade.
 
        capital_method e compound_fract são enviados para o C++ gerenciar
        o capital base dinamicamente durante o backtest.
 
        tick e tick_fin_val são injetados pelo Operation.py após esta chamada.
        """
        cap = capital if capital is not None else self.capital
        p   = self.sizing_params
        m   = self.sizing_method
 
        # sizing_method=signal → C++ lê lot_size do fast_pool via signal_refs por lado
        if m == "signal":
            return {
                "method":          "signal",
                "capital":         cap,
                "capital_method":  self.capital_method,
                "compound_fract":  self.compound_fract,
                # ref_long/ref_short resolvidos pelo Operation.py via signal_refs
            }
 
        base = {
            "method":         m,
            "capital":        cap,
            "capital_method": self.capital_method,
            "compound_fract": self.compound_fract,
            # compound_fract_series → injetado no indicators_pool como "compound_fract"
 
            "risk_pct":       p.get("risk_pct",       0.01),
            "risk_pct_min":   p.get("risk_pct_min",   0.001),
            "risk_pct_max":   p.get("risk_pct_max",   0.05),
            "fixed_lot":      p.get("fixed_lot",      1.0),
            "pct":            p.get("pct",             0.02),
            "kelly_weight":   p.get("kelly_weight",   0.25),
            "var_confidence": p.get("var_confidence", 0.95),
            "min_trades":     p.get("min_trades",     30),
        }
 
        # Referência de distância para risk_per_trade
        if m == "risk_per_trade":
            if self.dist_signal_ref:
                base["dist_ref"] = self.dist_signal_ref
            elif self.dist_fixed is not None:
                base["dist_fixed"] = self.dist_fixed
 
        return base
 
    # ─────────────────────────────────────────────────────────────────────────
    # Interface Portfolio Simulator — escalar com capital externo
    # ─────────────────────────────────────────────────────────────────────────
 
    def calc_lot_size(
        self,
        capital:           float,
        price:             float,
        dist_price:        float = 0.0,
        asset=None,
        trades:            Optional[List] = None,
        cumulative_profit: float = 0.0,   # lucro acumulado até o momento
    ) -> float:
        """
        Calcula lot_size escalar para Portfolio Simulator.
        Recebe capital externo de MMM/PMM.
        Aplica capital_method para ajustar o capital base antes do sizing.
        """
        # Ajusta capital base pelo capital_method
        capital_base = self._apply_capital_method(capital, cumulative_profit)
 
        m            = self.sizing_method
        p            = self.sizing_params
        tick         = getattr(asset, 'tick',         0.01) if asset else 0.01
        tick_fin_val = getattr(asset, 'tick_fin_val', 1.0)  if asset else 1.0
 
        if m == "neutral":
            return 1.0
 
        if m == "fixed":
            return float(p.get("fixed_lot", 1.0))
 
        if m == "risk_per_trade":
            dist       = dist_price if dist_price > 0.0 else (self.dist_fixed or tick)
            dist_ticks = dist / tick
            risk_usd   = capital_base * p.get("risk_pct", 0.01)
            lot        = risk_usd / (dist_ticks * tick_fin_val)
            return self._clamp_lot(lot, capital_base, price, tick_fin_val, p)
 
        if m == "pct_capital":
            lot = (capital_base * p.get("pct", 0.02)) / (price * tick_fin_val)
            return self._clamp_lot(lot, capital_base, price, tick_fin_val, p)
 
        if m == "kelly":
            lot = self._calc_kelly(capital_base, price, tick_fin_val, trades, p)
            return self._clamp_lot(lot, capital_base, price, tick_fin_val, p)
 
        if m == "var":
            lot = self._calc_var_lot(capital_base, price, tick_fin_val, trades, p)
            return self._clamp_lot(lot, capital_base, price, tick_fin_val, p)
 
        if m == "signal":
            return 1.0  # Portfolio Sim sem série disponível → neutro
 
        return 1.0
 
    # ─────────────────────────────────────────────────────────────────────────
    # Internos
    # ─────────────────────────────────────────────────────────────────────────
 
    def _apply_capital_method(self, capital: float, cumulative_profit: float) -> float:
        """
        Ajusta o capital base pelo capital_method.
 
        "fixed"          → capital fixo, sem compounding
        "compound_fract" → capital += cumulative_profit × compound_fract
                           no Portfolio Sim, usa compound_fract escalar
        """
        m = self.capital_method
 
        if m == "fixed":
            return capital
 
        if m in ("compound_fract", "compound"):
            fract = self.compound_fract
            return capital + cumulative_profit * fract
 
        return capital  # fallback
 
    def _clamp_lot(self, lot: float, capital: float, price: float,
                   tick_fin_val: float, p: dict) -> float:
        lot_min = (capital * p.get("risk_pct_min", 0.001)) / (price * tick_fin_val)
        lot_max = (capital * p.get("risk_pct_max", 0.05))  / (price * tick_fin_val)
        return float(max(lot_min, min(lot, lot_max)))
 
    def _calc_kelly(self, capital: float, price: float, tick_fin_val: float,
                    trades: Optional[List], p: dict) -> float:
        min_trades = int(p.get("min_trades", 30))
        if not trades or len(trades) < min_trades:
            return 1.0
        wins   = [t for t in trades if t.profit > 0]
        losses = [t for t in trades if t.profit <= 0]
        if not wins or not losses:
            return 1.0
        win_rate = len(wins) / len(trades)
        avg_win  = sum(t.profit for t in wins)       / len(wins)
        avg_loss = abs(sum(t.profit for t in losses) / len(losses))
        b        = avg_win / avg_loss
        kelly_f  = max(0.0, (win_rate * b - (1 - win_rate)) / b)
        kelly_f *= p.get("kelly_weight", 0.25)
        return (capital * kelly_f) / (price * tick_fin_val)
 
    def _calc_var_lot(self, capital: float, price: float, tick_fin_val: float,
                      trades: Optional[List], p: dict) -> float:
        min_trades = int(p.get("min_trades", 30))
        if not trades or len(trades) < min_trades:
            return 1.0
        returns  = np.array([t.profit for t in trades], dtype=np.float64)
        conf     = p.get("var_confidence", 0.95)
        var      = float(np.percentile(returns, (1 - conf) * 100))
        if var >= 0:
            return 1.0
        risk_usd = capital * p.get("risk_pct", 0.01)
        return risk_usd / (abs(var) * tick_fin_val)
 
    def __repr__(self):
        return (f"<StratMoneyManager sizing={self.sizing_method} "
                f"capital_method={self.capital_method} "
                f"compound_fract={self.compound_fract} "
                f"capital={self.capital}>")