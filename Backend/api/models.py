# ═══════════════════════════════════════════════════════════════════════════
# Backend/api/models.py — Pydantic schemas
# ═══════════════════════════════════════════════════════════════════════════
from pydantic import BaseModel
from typing import Optional
from datetime import date, datetime

class AssetGroupIn(BaseModel):
    name:           str
    market:         Optional[str]   = None
    tick:           Optional[float] = None
    tick_fin_val:   Optional[float] = None
    lot_min:        Optional[float] = None
    lot_step:       Optional[float] = None
    lot_max:        Optional[float] = None
    margin_rate:    Optional[float] = None
 
 
class AssetIn(BaseModel):
    name:           str
    type:           Optional[str]   = None
    market:         Optional[str]   = None
    group_id:       Optional[int]   = None
    tick:           Optional[float] = None
    tick_fin_val:   Optional[float] = None
    lot_min:        Optional[float] = None
    lot_step:       Optional[float] = None
    lot_max:        Optional[float] = None
    margin_rate:    Optional[float] = None
    swap_long:      Optional[float] = None
    swap_short:     Optional[float] = None
    data_path:      Optional[str]   = None
 
 
class AssetOut(AssetIn):
    id:         int
    created_at: Optional[datetime] = None
 
 
class OperationOut(BaseModel):
    id:           int
    name:         str
    date_start:   Optional[date]     = None
    date_end:     Optional[date]     = None
    status:       str
    n_models:     int
    n_strats:     int
    n_assets:     int
    n_param_sets: int
    created_at:   Optional[datetime] = None
 
 
class ParamSetOut(BaseModel):
    id:              int
    ps_name:         str
    param_json:      Optional[dict]  = None
    n_trades:        Optional[int]   = None
    total_pnl:       Optional[float] = None
    trades_path:     Optional[str]   = None
    pnl_matrix_path: Optional[str]   = None
    strat_name:      str
    backtest_mode:   Optional[str]   = None
    model_name:      str
    exec_tf:         Optional[str]   = None
    asset_name:      str
    market:          Optional[str]   = None
    operation_name:  str
    date_start:      Optional[date]  = None
    date_end:        Optional[date]  = None
    best_ps_name:    Optional[str]   = None
    best_wfe:        Optional[float] = None
    wf_json_path:    Optional[str]   = None
 
 
class PopulateResponse(BaseModel):
    operation_id: int
    message:      str














