# ═══════════════════════════════════════════════════════════════════════════
# Backend/api/routers/assets.py
# ═══════════════════════════════════════════════════════════════════════════
from fastapi import APIRouter, Depends, HTTPException
from api.deps import get_db
from api.models import AssetIn, AssetOut, AssetGroupIn
from db.Database import Database #type: ignore

router = APIRouter(prefix="/assets", tags=["assets"])

@router.get("", response_model=list[AssetOut])
def list_assets(db: Database = Depends(get_db)):
    # Returns all assets with resolved parameters (group inheritance applied)
    rows = db.query("""
        SELECT
            a.id, a.name, a.type, a.market,
            COALESCE(a.tick,         g.tick)         AS tick,
            COALESCE(a.tick_fin_val, g.tick_fin_val) AS tick_fin_val,
            COALESCE(a.lot_min,      g.lot_min)      AS lot_min,
            COALESCE(a.lot_step,     g.lot_step)     AS lot_step,
            COALESCE(a.lot_max,      g.lot_max)      AS lot_max,
            COALESCE(a.margin_rate,  g.margin_rate)  AS margin_rate,
            a.swap_long, a.swap_short, a.data_path,
            a.created_at
        FROM assets a
        LEFT JOIN asset_groups g ON g.id = a.group_id
        ORDER BY a.name
    """)
    return rows
 
 
@router.get("/{name}", response_model=AssetOut)
def get_asset(name: str, db: Database = Depends(get_db)):
    # Returns a single asset with resolved parameters.
    asset = db.get_asset(name)
    if not asset:
        raise HTTPException(404, f"Asset '{name}' not found.")
    return asset
 
 
@router.post("", response_model=AssetOut)
def create_or_update_asset(asset: AssetIn, db: Database = Depends(get_db)):
    # Creates or updates an asset. Returns the asset with resolved params.
    asset_id = db.upsert_asset(**asset.model_dump())
    return db.get_asset(asset.name)
 
 
@router.post("/groups", response_model=dict)
def create_or_update_asset_group(group: AssetGroupIn, db: Database = Depends(get_db)):
    # Creates or updates an asset group (shared params for multiple assets).
    group_id = db.upsert_asset_group(**group.model_dump())
    return {"id": group_id, "name": group.name}
 























