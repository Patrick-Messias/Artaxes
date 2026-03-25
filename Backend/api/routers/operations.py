# ═══════════════════════════════════════════════════════════════════════════
# Backend/api/routers/operations.py
# ═══════════════════════════════════════════════════════════════════════════
from fastapi import APIRouter, Depends, HTTPException
from api.deps import get_db
from api.models import OperationOut, PopulateResponse
from db.Database import Database # type: ignore

router = APIRouter(prefix="/operations", tags=["operations"])

@router.get("", response_model=list[OperationOut])
def list_operations(db: Database = Depends(get_db)):
    # Returns all operations with summary counts
    return db.get_operations()

@router.get("/{name}/summary")
def operation_summary(name: str, db: Database=Depends(get_db)):
    # Returns full hierarchy of an operation: models -> strats -> assets -> param_sets with metrics
    rows = db.get_param_sets(operation_name=name)
    if not rows:
        raise HTTPException(404, f"Operation '{name}' not found or has no results.")

    # Builds nested structure: model -> strat -> asset -> [param_sets]
    summary = {}
    for r in rows:
        m = r["model_name"]
        s = r["strat_name"]
        a = r["asset_name"]
        summary.setdefault(m, {}).setdefault(s, {}).setdefault(a, []).append({
            "id":          r["id"],
            "name":        r["ps_name"],
            "n_trades":    r["n_trades"],
            "total_pnl":   r["total_pnl"],
            "best_wfe":    r["best_wfe"],
            "best_ps":     r["best_ps_name"],
        })
    return summary

@router.post("/populate/{name}", response_model=PopulateResponse)
def populate_operation(name: str, db: Database = Depends(get_db)):
    try:
        # Chamamos a função do banco
        result = db.populate_operation(name)
        
        # Se o seu Database.py retornar um dicionário, pegamos o ID dele
        # Se retornar direto o ID, usamos direto. 
        # Vou assumir que o Database.py agora retorna o ID (int)
        
        if result is None:
             raise HTTPException(status_code=500, detail="Banco não retornou ID da operação")

        return PopulateResponse(
            operation_id=result if isinstance(result, int) else result.get("operation_id"),
            message=f"Operação '{name}' populada com sucesso!"
        )
    except Exception as e:
        print(f"  > [ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error populating operation: {str(e)}")

# @router.post("/populate/{name}", response_model=PopulateResponse)
# def populate_operation(name: str, db: Database = Depends(get_db)):
#     # Reads results/{name}/ and populates all metadata tables in DuckDB
#     # Safe to call multiple times, idempotent operation

#     try:
#         op_id = db.populate_operation(name)
#         return PopulateResponse(
#             operation_id=op_id,
#             message=f"Operation '{name}' populated sucessfully."
#         )
#     except FileNotFoundError as e:
#         raise HTTPException(404, str(e))
#     except Exception as e:
#         raise HTTPException(500, f"Error populating operation: {e}")
















