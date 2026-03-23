import numpy as np
import polars as pl
from dataclasses import dataclass, field
from typing import Optional, List, Literal
from MoneyManager import MoneyManager, MoneyManagerParams


# ─────────────────────────────────────────────────────────────────────────────
# Params
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StratMoneyManagerParams(MoneyManagerParams):

    # ── Position Sizing ──────────────────────────────────────────────────────
    sizing_method: Literal["neutral", "fixed", "risk_per_trade", "pct_capital", "kelly", "var", "signal"] = "neutral"
    # "neutral"         → lot = 1.0 (padrão — Portfolio Sim sobrescreve se quiser)
    # "fixed"           → lot fixo em sizing_params["fixed_lot"]              — C++
    # "risk_per_trade"  → capital × risk_pct / (dist_ticks × tick_fin_val)    — C++
    # "pct_capital"     → (capital × pct) / (price × tick_fin_val)            — C++
    # "kelly"           → kelly fraction usando histórico acumulado no C++
    # "var"             → VaR usando histórico acumulado no C++
    # "signal"          → lot vem de custom_lot_size_long/short (pl.Series)   — Python → pool
 
    sizing_params: dict = field(default_factory=lambda: {
        "fixed_lot":      1.0,    # para "fixed"
        "risk_pct":       0.01,   # para "risk_per_trade", "kelly" e "var"
        "pct":            0.02,   # para "pct_capital"
        "kelly_weight":   0.25,   # fração kelly (ex: 0.25 = quarter-kelly)
        "var_confidence": 0.95,   # para "var"
        "min_trades":     30,     # mínimo de trades para kelly/var serem ativados no C++
    })

    # ── Capital Management ───────────────────────────
    capital_method: Literal["fixed", "compound"] = "fixed"
    # "fixed"           → Always use initial capital, without compounding
    # "compound"        → Initial capital + (compound_fract=1.0 * profit)
 
    compound_fract: float = 1.0
    # Fração do lucro reinvestida no capital base
    # 0.0 = equivalente a "fixed"
    # 1.0 = compound total
    # 0.5 = half-compound
    # Se compound_fract_series definido → substitui este escalar barra a barra
 
    compound_fract_series: Optional[pl.Series] = None
    # pl.Series [0.0, 1.0] — substitui compound_fract barra a barra
    # Pode ser retornada em strat_signals com key "compound_fract_series"
    # ou definida diretamente aqui
    # C++ lê via fast_pool["compound_fract"]
 
    # ── Distância para risk_per_trade ─────────────────────────────────────────
    # Prioridade: dist_signal_ref → SL do trade → dist_fixed → tick do asset
    dist_signal_ref: Optional[pl.Series] = None
    # pl.Series com distância em pontos barra a barra
    # Ex: ATR, (high - low), spread × N, etc.
    # Pode ser retornada em strat_signals com key "dist_ref"
    # ou definida diretamente aqui
    # C++ lê via fast_pool["dist_ref"]
 
    dist_fixed: Optional[float] = None
    # Distância fixa em pontos como fallback quando dist_signal_ref não definido
 
    # ── Custom lot size via pl.Series (sizing_method="signal") ───────────────
    # Pode ser retornado em strat_signals com keys "custom_lot_size_long/short"
    # ou definido diretamente aqui
    custom_lot_size_long:  Optional[pl.Series] = None
    custom_lot_size_short: Optional[pl.Series] = None

# ─────────────────────────────────────────────────────────────────────────────
# StratMoneyManager
# ─────────────────────────────────────────────────────────────────────────────
 
class StratMoneyManager(MoneyManager):
    """
    Gerencia sizing de lote e capital base por Strat.
 
    capital_method:
        "fixed"    → capital fixo
        "compound" → capital += lucro × compound_fract
                     compound_fract_series substitui compound_fract barra a barra
 
    sizing_method:
        neutral/fixed/risk_per_trade/pct_capital/kelly/var → C++
        signal → Python via indicators_pool (custom_lot_size_long/short)
 
    dist_signal_ref (pl.Series):
        Distância para risk_per_trade — calculada em Python, injetada no pool
        Prioridade no C++: dist_ref pool → SL do trade → dist_fixed → tick
 
    Backtest individual:
        SMM definido  → lot_size real conforme sizing_method e capital_method
        SMM não definido → lot=1.0 neutro
 
    Portfolio Simulator:
        calc_lot_size(capital=capital_alocado) — mesmo SMM, capital externo
    """
 
    def __init__(self, params: StratMoneyManagerParams):
        super().__init__(params)
 
        self.sizing_method          = params.sizing_method
        self.sizing_params          = params.sizing_params
        self.capital_method         = params.capital_method
        self.compound_fract         = params.compound_fract
        self.compound_fract_series  = params.compound_fract_series
        self.dist_signal_ref        = params.dist_signal_ref
        self.dist_fixed             = params.dist_fixed
        self.custom_lot_size_long   = params.custom_lot_size_long
        self.custom_lot_size_short  = params.custom_lot_size_short
 
    # ─────────────────────────────────────────────────────────────────────────
    # Interface C++ — serializa método e params para sim_params
    # ─────────────────────────────────────────────────────────────────────────
 
    def to_sim_params(self, capital: Optional[float] = None) -> dict:
        """
        Serializa o SMM para dict que vai em sim_params["money_manager"].
        C++ lê em money_manager.cpp e calcula lot_size no open_trade.
 
        tick, tick_fin_val, min_lot, max_lot, lot_step
        são injetados pelo Operation.py após esta chamada.
        """
        cap = capital if capital is not None else self.capital
        p   = self.sizing_params
        m   = self.sizing_method
 
        base = {
            "method":         m,
            "capital":        cap,
            "capital_method": self.capital_method,   # "fixed" ou "compound"
            "compound_fract": self.compound_fract,   # substituído por série se presente no pool
 
            "risk_pct":       p.get("risk_pct",       0.01),
            "fixed_lot":      p.get("fixed_lot",       1.0),
            "pct":            p.get("pct",             0.02),
            "kelly_weight":   p.get("kelly_weight",   0.25),
            "var_confidence": p.get("var_confidence", 0.95),
            "min_trades":     p.get("min_trades",     30),
        }
 
        # dist_fixed como fallback para risk_per_trade
        # dist_signal_ref é injetado no indicators_pool pelo Operation.py
        # C++ resolve: fast_pool["dist_ref"] → SL do trade → dist_fixed → tick
        if m == "risk_per_trade" and self.dist_fixed is not None:
            base["dist_fixed"] = self.dist_fixed
 
        return base
 
    # ─────────────────────────────────────────────────────────────────────────
    # Interface Portfolio Simulator — escalar com capital externo
    # ─────────────────────────────────────────────────────────────────────────
 
    def calc_lot_size(
        self,
        capital:           float,
        price:             float,
        dist_price:        float = 0.0,   # distância em preço (ATR, SL, etc.)
        asset=None,
        trades:            Optional[List] = None,
        cumulative_profit: float = 0.0,   # lucro acumulado em $ para compound
    ) -> float:
        """
        Calcula lot_size escalar para Portfolio Simulator.
        Recebe capital externo de MMM/PMM.
        Aplica capital_method antes do sizing.
        Arredondamento via min_lot/lot_step fica no Portfolio Simulator.
        """
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
            # Prioridade: dist_price arg → dist_fixed → tick
            dist       = dist_price if dist_price > 0.0 else (self.dist_fixed or tick)
            dist_ticks = dist / tick
            if dist_ticks <= 0.0: return 1.0
            return (capital_base * p.get("risk_pct", 0.01)) / (dist_ticks * tick_fin_val)
 
        if m == "pct_capital":
            if price <= 0.0 or tick_fin_val <= 0.0: return 1.0
            return (capital_base * p.get("pct", 0.02)) / (price * tick_fin_val)
 
        if m == "kelly":
            return self._calc_kelly(capital_base, price, tick_fin_val, trades, p)
 
        if m == "var":
            return self._calc_var_lot(capital_base, price, tick_fin_val, trades, p)
 
        if m == "signal":
            return 1.0  # Portfolio Sim sem série → neutro
 
        return 1.0
 
    # ─────────────────────────────────────────────────────────────────────────
    # Internos
    # ─────────────────────────────────────────────────────────────────────────
 
    def _apply_capital_method(self, capital: float, cumulative_profit: float) -> float:
        """
        "fixed"    → capital constante
        "compound" → capital + cumulative_profit × compound_fract
                     cumulative_profit deve ser em $ no Portfolio Simulator
        """
        if self.capital_method == "fixed":
            return capital
        # compound
        return capital + cumulative_profit * self.compound_fract
 
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
        returns = np.array([t.profit for t in trades], dtype=np.float64)
        conf    = p.get("var_confidence", 0.95)
        var     = float(np.percentile(returns, (1 - conf) * 100))
        if var >= 0:
            return 1.0
        return (capital * p.get("risk_pct", 0.01)) / (abs(var) * tick_fin_val)
 
    def __repr__(self):
        return (f"<StratMoneyManager sizing={self.sizing_method} "
                f"capital_method={self.capital_method} "
                f"compound_fract={self.compound_fract} "
                f"capital={self.capital}>")