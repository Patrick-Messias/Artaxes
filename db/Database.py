"""
Database.py - DuckDB connection, schema creation and helpers

Usage:
    from db.database import Database
    db = Database()
    db.populate_operation("operation_test)
"""

import polars as pl, duckdb, json, os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH      = Path(os.getenv("DB_PATH",      "./db/art.duckdb"))
RESULTS_PATH = Path(os.getenv("RESULTS_PATH", "./results"))
SCHEMA_PATH  = Path(os.path.dirname(__file__)) / "schema.sql"

class Database:
    # Manages DuckDB connection and all database operations
    # Single instance per process, DuckDB supports multiple readers
    # but only one writer at time
 
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.db_path))
        self._create_schema()

    def _create_schema(self):
        # Creates all tables and views from schema.sql if the don't exist
        with open(SCHEMA_PATH, 'r') as f:
            sql = f.read()
        self.con.executescript(sql)

    def close(self):
        self.con.close()

    def __enter__(self):
        return self
    
    def __exit__(self, *_):
        self.close()

    # ── Asset helpers ─────────────────────────────────────────────────────────

    def upsert_asset_group(self, name: str, market: str=None, **params) -> int:
        # Insterts or updates an asset group, returns group id

        existing = self.con.execute(
            "SELECT id FROM asset_groups WHERE name = ?", [name]
        ).fetchone()

        if existing:
            self.con.execute("""
                UPDATE asset_groups SET
                    market       = COALESCE(?, market),
                    tick         = COALESCE(?, tick),
                    tick_fin_val = COALESCE(?, tick_fin_val),
                    lot_min      = COALESCE(?, lot_min),
                    lot_step     = COALESCE(?, lot_step),
                    lot_max      = COALESCE(?, lot_max),
                    margin_rate  = COALESCE(?, margin_rate)
                WHERE name = ?
            """, [
                market,
                params.get("tick"),
                params.get("tick_fin_val"),
                params.get("lot_min"),
                params.get("lot_step"),
                params.get("lot_max"),
                params.get("margin_rate"),
                name
            ])
            return existing[0]

        self.con.execute("""
            INSERT INTO asset_groups (name, market, tick, tick_fin_val,
                                      lot_min, lot_step, lot_max, margin_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            name, market,
            params.get("tick"),
            params.get("tick_fin_val"),
            params.get("lot_min"),
            params.get("lot_step"),
            params.get("lot_max"),
            params.get("margin_rate"),
        ])
        return self.con.execute(
            "SELECT id FROM asset_groups WHERE name = ?", [name]
        ).fetchone()[0]

    def upsert_asset(self, name: str, asset_type: str=None,
                        market: str=None, group_id: int=None,
                        data_path: str=None, **params) -> int:
        # Inserts or updates an asset, returns asset id
        # Asset-level params override group params when not NULL

        existing = self.con.execute(
            "SELECT id FROM assets WHERE name = ?", [name]
        ).fetchone()

        if existing:
            self.con.execute("""
                UPDATE assets SET
                    group_id     = COALESCE(?, group_id),
                    type         = COALESCE(?, type),
                    market       = COALESCE(?, market),
                    tick         = COALESCE(?, tick),
                    tick_fin_val = COALESCE(?, tick_fin_val),
                    lot_min      = COALESCE(?, lot_min),
                    lot_step     = COALESCE(?, lot_step),
                    lot_max      = COALESCE(?, lot_max),
                    margin_rate  = COALESCE(?, margin_rate),
                    swap_long    = COALESCE(?, swap_long),
                    swap_short   = COALESCE(?, swap_short),
                    data_path    = COALESCE(?, data_path)
                WHERE name = ?
            """, [
                group_id, asset_type, market,
                params.get("tick"), params.get("tick_fin_val"),
                params.get("lot_min"), params.get("lot_step"),
                params.get("lot_max"), params.get("margin_rate"),
                params.get("swap_long"), params.get("swap_short"),
                data_path, name
            ])
            return existing[0]
 
        self.con.execute("""
            INSERT INTO assets (group_id, name, type, market, tick, tick_fin_val,
                                lot_min, lot_step, lot_max, margin_rate,
                                swap_long, swap_short, data_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            group_id, name, asset_type, market,
            params.get("tick"), params.get("tick_fin_val"),
            params.get("lot_min"), params.get("lot_step"),
            params.get("lot_max"), params.get("margin_rate"),
            params.get("swap_long"), params.get("swap_short"),
            data_path
        ])
        return self.con.execute(
            "SELECT id FROM assets WHERE name = ?", [name]
        ).fetchone()[0]

    def get_asset(self, name: str) -> Optional[dict]:
        # Retuns asset with resolved params - merges group defaults
        # with asset-level overrides. Asset fields take priority over group
        row = self.con.execute("""
            SELECT
                a.id, a.name, a.type, a.market,
                COALESCE(a.tick,         g.tick)         AS tick,
                COALESCE(a.tick_fin_val, g.tick_fin_val) AS tick_fin_val,
                COALESCE(a.lot_min,      g.lot_min)      AS lot_min,
                COALESCE(a.lot_step,     g.lot_step)     AS lot_step,
                COALESCE(a.lot_max,      g.lot_max)      AS lot_max,
                COALESCE(a.margin_rate,  g.margin_rate)  AS margin_rate,
                a.swap_long, a.swap_short, a.data_path
            FROM assets a
            LEFT JOIN asset_groups g ON g.id = a.group_id
            WHERE a.name = ?
        """, [name]).fetchone()
 
        if not row:
            return None
 
        cols = ["id", "name", "type", "market", "tick", "tick_fin_val",
                "lot_min", "lot_step", "lot_max", "margin_rate",
                "swap_long", "swap_short", "data_path"]
        return dict(zip(cols, row))

    # ── Populate from results/ ────────────────────────────────────────────────

    def populate_operation(self, operation_name: str) -> int:
        # Reads results/{operation_name}/ and populates all metadata tables
        # Idempotent - safe to call multiple times, updates existing records
        # Returns the operation id

        # Structure expected:
        #       results/{op_name}/
        #               meta/operation_meta.json
        #               trades/{model}/{strat}/{asset}/
        #                       {ps_name}.parquet
        #                       pnl_matrix.parquet
        #                       lot_matrix.parquet
        #                       all_wf_results.json (optional)

        op_path = RESULTS_PATH / operation_name
        meta_path = op_path / "meta" / "operation_meta.json"
 
        if not op_path.exists():
            raise FileNotFoundError(f"Operation path not found: {op_path}")
 
        # Lê meta da operation
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
 
        # Upsert operation
        op_id = self._upsert_operation(
            name=operation_name,
            date_start=meta.get("date_start"),
            date_end=meta.get("date_end"),
            op_timeframe=meta.get("operation_timeframe"),
            results_path=str(op_path),
        )
 
        trades_path = op_path / "trades"
        if not trades_path.exists():
            print(f"   > No trades found for {operation_name}")
            return op_id
 
        # Itera models → strats → assets
        for model_name in sorted(trades_path.iterdir()):
            if not model_name.is_dir(): continue
 
            # Lê exec_tf do meta se disponível
            exec_tf = meta.get("models", {}).get(model_name.name, {}).get("exec_tf")
            model_id = self._upsert_model(op_id, model_name.name, exec_tf)
 
            for strat_dir in sorted(model_name.iterdir()):
                if not strat_dir.is_dir(): continue
 
                strat_id = self._upsert_strat(model_id, strat_dir.name)
 
                for asset_dir in sorted(strat_dir.iterdir()):
                    if not asset_dir.is_dir(): continue
 
                    asset_name = asset_dir.name
                    # Garante que o asset existe (sem params — usuário cadastra depois)
                    asset_id = self._get_or_create_asset(asset_name)
 
                    # Liga asset ao model
                    self._upsert_model_asset(model_id, asset_id)
 
                    # Popula param_sets — todos os .parquet exceto os reservados
                    reserved = {"pnl_matrix.parquet", "lot_matrix.parquet", "wfm_raw.parquet"}
                    pnl_matrix_path = asset_dir / "pnl_matrix.parquet"
                    lot_matrix_path = asset_dir / "lot_matrix.parquet"
 
                    for ps_file in sorted(asset_dir.glob("*.parquet")):
                        if ps_file.name in reserved: continue
 
                        ps_name = ps_file.stem
                        n_trades, total_pnl = self._read_trade_metrics(ps_file)
 
                        self._upsert_param_set(
                            strat_id=strat_id,
                            asset_id=asset_id,
                            name=ps_name,
                            trades_path=str(ps_file),
                            pnl_matrix_path=str(pnl_matrix_path) if pnl_matrix_path.exists() else None,
                            lot_matrix_path=str(lot_matrix_path) if lot_matrix_path.exists() else None,
                            n_trades=n_trades,
                            total_pnl=total_pnl,
                        )
 
                    # WF results
                    wf_json = asset_dir / "all_wf_results.json"
                    if wf_json.exists():
                        self._upsert_wf_result(strat_id, asset_id, str(wf_json))
 
        print(f"   > [DB] Operation '{operation_name}' populated (id={op_id})")
        return op_id

    # ── Internal upserts ─────────────────────────────────────────────────────

    def _upsert_operation(self, name: str, date_start=None, date_end=None, op_timeframe=None, results_path=None) -> int:
        existing = self.con.execute(
            "SELECT id FROM operations WHERE name = ?", [name]
        ).fetchone()

        if existing:
            self.con.execute("""
                UPDATE operations SET
                    date_start   = COALESCE(?, date_start),
                    date_end     = COALESCE(?, date_end),
                    op_timeframe = COALESCE(?, op_timeframe),
                    results_path = COALESCE(?, results_path),
                    status       = 'done'
                WHERE name = ?
            """, [date_start, date_end, op_timeframe, results_path, name])
            return existing[0]
 
        self.con.execute("""
            INSERT INTO operations (name, date_start, date_end, op_timeframe,
                                    results_path, status)
            VALUES (?, ?, ?, ?, ?, 'done')
        """, [name, date_start, date_end, op_timeframe, results_path])
        return self.con.execute(
            "SELECT id FROM operations WHERE name = ?", [name]
        ).fetchone()[0]

    def _upsert_model(self, op_id: int, name: str, exec_tf: str=None) -> int:
        existing = self.con_execute(
            "SELECT id FROM models WHERE operation_id = ? AND name = ?",
            [op_id, name]
        ).fetchone()

        if existing: 
            return existing[0]

        self.con.execute(
            "INSERT INTO models (operation_id, name, exec_tf) VALUES (?, ?, ?)",
            [op_id, name, exec_tf]
        )
        return self.con.execute(
            "SELECT id FROM models WHERE operation_id = ? AND name = ?",
            [op_id, name]
        ).fetchone()[0]

    def _upsert_strat(self, model_id: int, name: str) -> int:
        existing = self.con.execute(
            "SELECT id FROM strats WHERE model_id = ? AND name = ?",
            [model_id, name]
        ).fetchone()
 
        if existing:
            return existing[0]
 
        self.con.execute(
            "INSERT INTO strats (model_id, name) VALUES (?, ?)",
            [model_id, name]
        )
        return self.con.execute(
            "SELECT id FROM strats WHERE model_id = ? AND name = ?",
            [model_id, name]
        ).fetchone()[0]
 
    def _get_or_create_asset(self, name: str) -> int:
        existing = self.con.execute(
            "SELECT id FROM assets WHERE name = ?", [name]
        ).fetchone()
 
        if existing:
            return existing[0]
 
        self.con.execute(
            "INSERT INTO assets (name) VALUES (?)", [name]
        )
        return self.con.execute(
            "SELECT id FROM assets WHERE name = ?", [name]
        ).fetchone()[0]
 
    def _upsert_model_asset(self, model_id: int, asset_id: int):
        existing = self.con.execute(
            "SELECT 1 FROM model_assets WHERE model_id = ? AND asset_id = ?",
            [model_id, asset_id]
        ).fetchone()
 
        if not existing:
            self.con.execute(
                "INSERT INTO model_assets (model_id, asset_id) VALUES (?, ?)",
                [model_id, asset_id]
            )
 
    def _upsert_param_set(self, strat_id: int, asset_id: int, name: str,
                           trades_path: str = None, pnl_matrix_path: str = None,
                           lot_matrix_path: str = None, n_trades: int = None,
                           total_pnl: float = None) -> int:
        existing = self.con.execute(
            "SELECT id FROM param_sets WHERE strat_id = ? AND asset_id = ? AND name = ?",
            [strat_id, asset_id, name]
        ).fetchone()
 
        if existing:
            self.con.execute("""
                UPDATE param_sets SET
                    trades_path     = COALESCE(?, trades_path),
                    pnl_matrix_path = COALESCE(?, pnl_matrix_path),
                    lot_matrix_path = COALESCE(?, lot_matrix_path),
                    n_trades        = COALESCE(?, n_trades),
                    total_pnl       = COALESCE(?, total_pnl)
                WHERE id = ?
            """, [trades_path, pnl_matrix_path, lot_matrix_path,
                  n_trades, total_pnl, existing[0]])
            return existing[0]
 
        self.con.execute("""
            INSERT INTO param_sets (strat_id, asset_id, name, trades_path,
                                    pnl_matrix_path, lot_matrix_path,
                                    n_trades, total_pnl)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [strat_id, asset_id, name, trades_path,
              pnl_matrix_path, lot_matrix_path, n_trades, total_pnl])
        return self.con.execute(
            "SELECT id FROM param_sets WHERE strat_id = ? AND asset_id = ? AND name = ?",
            [strat_id, asset_id, name]
        ).fetchone()[0]
 
    def _upsert_wf_result(self, strat_id: int, asset_id: int,
                           json_path: str) -> int:
        # Reads best_ps_name and best_wfe from JSON to cache
        best_ps, best_wfe = None, None
        try:
            with open(json_path) as f:
                wf = json.load(f)
            meta = wf.get("__meta__", {})
            res_key = meta.get("res_key", "wfe")
            # Encontra melhor config
            candidates = {k: v for k, v in wf.items() if k != "__meta__" and isinstance(v, dict)}
            if candidates:
                best_k = max(candidates, key=lambda k: candidates[k].get(res_key, -float("inf")))
                best_wfe = candidates[best_k].get(res_key)
                best_ps = candidates[best_k].get("selected_ps")
        except Exception:
            pass
 
        existing = self.con.execute(
            "SELECT id FROM wf_results WHERE strat_id = ? AND asset_id = ?",
            [strat_id, asset_id]
        ).fetchone()
 
        if existing:
            self.con.execute("""
                UPDATE wf_results SET
                    json_path    = ?,
                    best_ps_name = COALESCE(?, best_ps_name),
                    best_wfe     = COALESCE(?, best_wfe)
                WHERE id = ?
            """, [json_path, best_ps, best_wfe, existing[0]])
            return existing[0]
 
        self.con.execute("""
            INSERT INTO wf_results (strat_id, asset_id, json_path,
                                    best_ps_name, best_wfe)
            VALUES (?, ?, ?, ?, ?)
        """, [strat_id, asset_id, json_path, best_ps, best_wfe])
        return self.con.execute(
            "SELECT id FROM wf_results WHERE strat_id = ? AND asset_id = ?",
            [strat_id, asset_id]
        ).fetchone()[0]
 
    def _read_trade_metrics(self, parquet_path: Path):
        """Reads n_trades and total_pnl from a trades parquet file."""
        try:
            df = pl.read_parquet(str(parquet_path), columns=["profit"])
            return len(df), float(df["profit"].sum())
        except Exception:
            return None, None

    # ── Query helpers — used by FastAPI ──────────────────────────────────

    def get_operations(self) -> list[dict]:
        # Returns all operations with summary counts.
        rows = self.con.execute("SELECT * FROM v_operations_summary").fetchall()
        cols = ["id", "name", "date_start", "date_end", "status",
                "n_models", "n_strats", "n_assets", "n_param_sets", "created_at"]
        return [dict(zip(cols, r)) for r in rows]
 
    def get_param_sets(self, operation_name: str = None,
                        asset_name: str = None,
                        strat_name: str = None) -> list[dict]:
        # Returns param_sets with full context (operation, model, strat, asset).
        # Filters are optional and combinable.

        filters, params = [], []
        if operation_name:
            filters.append("o.name = ?")
            params.append(operation_name)
        if asset_name:
            filters.append("a.name = ?")
            params.append(asset_name)
        if strat_name:
            filters.append("s.name = ?")
            params.append(strat_name)
 
        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        rows = self.con.execute(
            f"SELECT * FROM v_param_sets_full {where}", params
        ).fetchall()
 
        cols = ["id", "ps_name", "param_json", "n_trades", "total_pnl",
                "trades_path", "pnl_matrix_path", "strat_name", "backtest_mode",
                "model_name", "exec_tf", "asset_name", "market",
                "operation_name", "date_start", "date_end",
                "best_ps_name", "best_wfe", "wf_json_path"]
        return [dict(zip(cols, r)) for r in rows]
 
    def get_pnl_matrix(self, operation_name: str,
                        model_name: str, strat_name: str,
                        asset_name: str) -> Optional[pl.DataFrame]:
        # Reads pnl_matrix.parquet directly via DuckDB — no data copy.
        # Returns None if not found.

        row = self.con.execute("""
            SELECT DISTINCT ps.pnl_matrix_path
            FROM param_sets ps
            JOIN strats s ON s.id = ps.strat_id
            JOIN models m ON m.id = s.model_id
            JOIN operations o ON o.id = m.operation_id
            JOIN assets a ON a.id = ps.asset_id
            WHERE o.name = ? AND m.name = ? AND s.name = ? AND a.name = ?
              AND ps.pnl_matrix_path IS NOT NULL
            LIMIT 1
        """, [operation_name, model_name, strat_name, asset_name]).fetchone()
 
        if not row or not row[0]:
            return None
 
        return pl.read_parquet(row[0])
 
    def get_trades(self, ps_id: int) -> Optional[pl.DataFrame]:
        # Reads trades parquet for a specific param_set.

        row = self.con.execute(
            "SELECT trades_path FROM param_sets WHERE id = ?", [ps_id]
        ).fetchone()
 
        if not row or not row[0]:
            return None
 
        return pl.read_parquet(row[0])
 
    def query(self, sql: str, params: list = None) -> list[dict]:
        # Raw SQL query — for custom frontend queries.
        # Returns list of dicts.
    
        result = self.con.execute(sql, params or [])
        cols = [d[0] for d in result.description]
        return [dict(zip(cols, row)) for row in result.fetchall()]

# ─────────────────────────────────────────────────────────────────────────





