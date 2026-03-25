# ═══════════════════════════════════════════════════════════════════════════
# Backend/api/routers/results.py
# ═══════════════════════════════════════════════════════════════════════════
from fastapi import APIRouter, Depends, HTTPException, Query
from api.deps import get_db
from api.models import ParamSetOut
from db.Database import Database #type: ignore
import polars as pl, json, os

router = APIRouter(prefix="/results", tags=["results"])

@router.get("/param-sets", response_model=list[ParamSetOut])
def list_param_sets(
    operation: str=Query(None),
    asset: str=Query(None),
    strat: str=Query(None),
    db: Database=Depends(get_db)
):
    return db.get_param_sets(operation_name=operation, asset_name=asset, strat_name=strat)

# ── NOVO: Endpoint para listar as combinações de Walk Forward ──────────────
@router.get("/param-sets/{ps_id}/wf-combos")
def get_wf_combos(ps_id: int, db: Database=Depends(get_db)):
    """Retorna as chaves do JSON (ex: IS1_OS1_ST1) para o Sidebar expandir"""
    row = db.query("SELECT wf_json_path FROM param_sets WHERE id = ?", [ps_id])
    if not row or not row[0].get("wf_json_path"):
        return []

    path = row[0]["wf_json_path"]
    if not os.path.exists(path):
        return []

    with open(path, 'r') as f:
        data = json.load(f)
    return list(data.keys())

# ── NOVO: Endpoint para pegar a Equity de um Walk Forward específico ───────
@router.get("/param-sets/{ps_id}/wf-equity")
def get_wf_equity(ps_id: int, combo_name: str, db: Database=Depends(get_db)):
    """Retorna a curva de equity pronta do JSON para o gráfico"""
    row = db.query("SELECT wf_json_path FROM param_sets WHERE id = ?", [ps_id])
    if not row or not row[0].get("wf_json_path"):
        raise HTTPException(404, "Caminho do JSON não encontrado")

    path = row[0]["wf_json_path"]
    with open(path, 'r') as f:
        data = json.load(f)
    
    combo = data.get(combo_name)
    if not combo:
        raise HTTPException(404, "Combinação não encontrada no arquivo")

    return combo.get("equity", [])



@router.get("/param-sets/{ps_id}/trades")
def get_trades(ps_id: int, db: Database=Depends(get_db)):
    df = db.get_trades(ps_id)
    if df is None: raise HTTPException(404, "Trades não encontrados")
    return df.to_dicts()

@router.get("/param-sets/{ps_id}/pnl-matrix")
def get_pnl_matrix(ps_id: int, db: Database=Depends(get_db)):
    # Returns all_matrix (daily returns) for the asset of a param_set
    # All param_sets of the same Asset/Strat share the same matrix

    row = db.query("""
        SELECT o.name, m.name, s.name, a.name
        FROM param_sets ps
        JOIN strats s ON s.id = ps.strat_id
        JOIN models m ON m.id = s.model_id
        JOIN operations o ON o.id = m.operation_id
        JOIN assets a ON a.id = ps.asset_id
        WHERE ps.id = ?
    """, [ps_id])

    if not row:
        raise HTTPException(404, f"Param set id={ps_id} not found")

    r = row[0]
    df = db.get_pnl_matrix(r["name"], r["name_1"], r["name_2"], r["name_3"])
    if df is None:
        raise HTTPException(404, "PnL matrix not found")

    # Converts datetime to string to serialize JSON
    return df.with_columns(
        pl.col("ts").cast(pl.Utf8)
    ).to_dicts()

@router.post("/basket/equity-curve")
def basket_equity_curve(ps_ids: list[int], db: Database = Depends(get_db)):
    if not ps_ids:
        raise HTTPException(400, "No param_set ids provided")

    frames = []
    for ps_id in ps_ids:
        df = db.get_trades(ps_id)
        if df is None or df.is_empty(): continue

        cols = df.columns
        date_col = "ts" if "ts" in cols else ("exit_datetime" if "exit_datetime" in cols else None)
        value_col = next((c for c in cols if c != date_col), None)

        if not date_col or not value_col: continue

        temp_df = df.select([
            pl.col(date_col).alias("datetime"),
            pl.col(value_col).cast(pl.Float64).alias(f"ps_{ps_id}")
        ])

        try:
            temp_df = temp_df.with_columns(pl.col("datetime").str.to_datetime("%Y%m%d %H%M%S"))
        except: pass 

        temp_df = temp_df.group_by("datetime").agg(pl.col(f"ps_{ps_id}").sum())
        frames.append(temp_df)

    if not frames:
        raise HTTPException(404, "No trades found for selected param_sets")

    from functools import reduce
    combined = reduce(
        lambda a, b: a.join(b, on="datetime", how="full", coalesce=True),
        frames
    ).sort("datetime").fill_null(0.0)

    ps_cols = [c for c in combined.columns if c != "datetime"]
    combined = combined.with_columns(
        pl.sum_horizontal(ps_cols).alias("total_pnl")
    ).with_columns(
        pl.col("total_pnl").cum_sum().alias("cumulative_pnl")
    )

    return combined.with_columns(pl.col("datetime").cast(pl.Utf8)).to_dicts()

# @router.post("/basket/equity-curve")
# def basket_equity_curve(
#     ps_ids: list[int],
#     db: Database = Depends(get_db)
# ):
#     if not ps_ids:
#         raise HTTPException(400, "No param_set ids provided")

#     frames = []
#     for ps_id in ps_ids:
#         df = db.get_trades(ps_id)
#         if df is None or df.is_empty():
#             continue

#         # 1. Identifica as colunas dinamicamente
#         cols = df.columns
#         date_col = "ts" if "ts" in cols else ("exit_datetime" if "exit_datetime" in cols else None)
        
#         # Procura a coluna de valor (que não seja a de data)
#         value_col = next((c for c in cols if c != date_col), None)

#         if not date_col or not value_col:
#             continue

#         # 2. Processa o DataFrame de forma robusta
#         temp_df = df.select([
#             pl.col(date_col).alias("datetime"),
#             pl.col(value_col).cast(pl.Float64).alias(f"ps_{ps_id}")
#         ])

#         # Se a data for string (YYYYMMDD...), converte. Se for timestamp, ignora erro.
#         try:
#             temp_df = temp_df.with_columns(
#                 pl.col("datetime").str.to_datetime("%Y%m%d %H%M%S")
#             )
#         except:
#             pass 

#         # Agrupa por data (caso haja mais de um trade no mesmo segundo)
#         temp_df = temp_df.group_by("datetime").agg(
#             pl.col(f"ps_{ps_id}").sum()
#         )
        
#         frames.append(temp_df)

#     if not frames:
#         raise HTTPException(404, "No trades found for selected param_sets")

#     # 3. Une todos os DataFrames (Inner Join ou Full Join)
#     from functools import reduce
#     combined = reduce(
#         lambda a, b: a.join(b, on="datetime", how="full", coalesce=True),
#         frames
#     ).sort("datetime").fill_null(0.0)

#     # 4. Calcula o PnL Total e Acumulado
#     ps_cols = [c for c in combined.columns if c != "datetime"]
#     combined = combined.with_columns(
#         pl.sum_horizontal(ps_cols).alias("total_pnl")
#     ).with_columns(
#         pl.col("total_pnl").cum_sum().alias("cumulative_pnl")
#     )

#     return combined.with_columns(
#         pl.col("datetime").cast(pl.Utf8)
#     ).to_dicts()

# @router.post("/basket/equity-curve")
# def basket_equity_curve(
#     ps_ids: list[int],
#     db: Database = Depends(get_db)
# ):
#     # Returns combined equity curve for a basket of param_sets
#     # Sums daily PnL across all selected param_sets aligned by datetime

#     if not ps_ids:
#         raise HTTPException(400, "No param_set ids provided")

#     frames = []
#     for ps_id in ps_ids:
#         df = db.get_trades(ps_id)
#         if df is None or df.is_empty():
#             continue
 
#         df = df.filter(pl.col("exit_datetime").is_not_null())
#         df = df.select([
#             pl.col("exit_datetime").str.to_datetime("%Y%m%d %H%M%S").alias("datetime"),
#             pl.col("profit").cast(pl.Float64).alias(f"ps_{ps_id}")
#         ]).group_by("datetime").agg(
#             pl.col(f"ps_{ps_id}").sum()
#         )
#         frames.append(df)
 
#     if not frames:
#         raise HTTPException(404, "No trades found for selected param_sets")
 
#     from functools import reduce
#     combined = reduce(
#         lambda a, b: a.join(b, on="datetime", how="full", coalesce=True),
#         frames
#     ).sort("datetime").fill_null(0.0)
 
#     # Sums total and accumulate
#     ps_cols = [c for c in combined.columns if c != "datetime"]
#     combined = combined.with_columns(
#         pl.sum_horizontal(ps_cols).alias("total_pnl")
#     ).with_columns(
#         pl.col("total_pnl").cum_sum().alias("cumulative_pnl")
#     )
 
#     return combined.with_columns(
#         pl.col("datetime").cast(pl.Utf8)
#     ).to_dicts()
 






