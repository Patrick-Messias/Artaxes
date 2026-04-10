import polars as pl, json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

class Storage:
    def __init__(self, base_path: str = "Backend/results"):
        self.base_path = Path(base_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────────────────────────────────

    def _asset_path(self, op: str, model: str, strat: str, asset: str) -> Path:
        return self.base_path / op / model / strat / asset
    
    def save_matrix_data(self, op: str, model: str, strat: str, asset: str, matrix_df: pl.DataFrame):
        # Salva a matriz de resultados no formato vertical (Long Format).
        # Contém ts, pnl, lot, mae, mfe e trade_id.
        
        if matrix_df is None or matrix_df.is_empty():
            return

        path = self._asset_path(op, model, strat, asset) / "matrix"
        path.mkdir(parents=True, exist_ok=True)

        # Salvamos em um arquivo único 'trades_matrix.parquet'
        file_path = path / "trades_matrix.parquet"
        
        matrix_df.write_parquet(
            file_path,
            compression="zstd",
            statistics=True
        )
        print(f"      > [Storage] Saved Matrix ({len(matrix_df)} rows) to {file_path}")

    def save_batch_trades(self, op: str, model: str, strat: str, asset: str, df_all_trades: pl.DataFrame):
        # Salva todos os trades detalhados de todos os parsets em um único arquivo.

        if df_all_trades is None or df_all_trades.is_empty():
            return

        path = self._asset_path(op, model, strat, asset) / "trades"
        path.mkdir(parents=True, exist_ok=True)
        
        file_path = path / "trades.parquet"
        
        df_all_trades.write_parquet(
            file_path, 
            compression="zstd", 
            statistics=True 
        )
        print(f"      > [Storage] Saved {len(df_all_trades)} trades to {file_path}")

    def save_walkforward(self, op, model, strat, asset, wf_id, all_runs):
        path = self._asset_path(op, model, strat, asset) / "wfm"
        path.mkdir(parents=True, exist_ok=True)

        frames = []

        for run in all_runs:
            if "os_curve" in run and isinstance(run["os_curve"], pl.DataFrame):
                df = run["os_curve"].clone()

                if "ts" in df.columns:
                    df = df.rename({"ts": "datetime"})

                df = df.with_columns([
                    pl.lit(run.get("best_param", "")).alias("best_param"),
                    pl.lit(wf_id).alias("wf_id")   # 🔥 chave
                ])

                frames.append(df)

        if not frames:
            return

        new_df = pl.concat(frames)

        file_path = path / "wf.parquet"

        # 🔥 append (múltiplos wf dentro de 1 arquivo)
        if file_path.exists():
            old = pl.read_parquet(file_path)
            new_df = pl.concat([old, new_df])

        new_df.sort("datetime").write_parquet(file_path)

    def save_operation_meta(self, op_name: str, meta_dict: dict):
        path = self.base_path / op_name / "operation_meta.json"
        with open(path, "w") as f:
            json.dump(meta_dict, f, indent=4)

    # ─────────────────────────────────────────────────────────────────────────
    # Load
    # ─────────────────────────────────────────────────────────────────────────

    def load_operation_meta(self, op_name: str) -> dict:
        path = self.base_path / op_name / "operation_meta.json"
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return {}

    def load_meta(self, operation_name: str) -> dict:
        meta_file = self.base_path / operation_name / "operation_meta.json"
        if not meta_file.exists():
            return {}
        with open(meta_file, "r") as f:
            return json.load(f)

    def list_operations(self) -> list:
        """Lista todas as operations salvas."""
        if not self.base_path.exists():
            return []
        return [d.name for d in self.base_path.iterdir() if d.is_dir()]

    def list_param_sets(self, operation_name: str, model: str, strat: str, asset: str) -> list:
        """Lista todos os ps_names salvos para um model/strat/asset."""
        path = self.base_path / operation_name / "trades" / model / strat / asset
        if not path.exists():
            return []
        return [f.stem for f in sorted(path.glob("*.parquet"))]

    # Reads trades and trades_matrix, generates a unified vertical structure with all parsets mixed in
    def load(self, op: str, model: str, strat: str, asset: str) -> Dict[str, Any]:

        asset_path = self._asset_path(op, model, strat, asset)
        asset_data = {}

        trades_path = asset_path / "trades" / "trades.parquet"
        matrix_path = asset_path / "matrix" / "trades_matrix.parquet"

        trades_df = pl.read_parquet(trades_path) if trades_path.exists() else None
        matrix_df = pl.read_parquet(matrix_path) if matrix_path.exists() else None

        if trades_df is not None and not trades_df.is_empty():
            asset_data["trades"] = trades_df
            
            # Gera a timeline imediatamente ao carregar
            timeline_df = self._build_timeline(trades_df, matrix_df)
            asset_data["timeline"] = timeline_df

        # ── WFM (Resultados Salvos do Walkforward) ────────────────────
        wf_file = asset_path / "wfm" / "wf.parquet"
        if wf_file.exists():
            asset_data["wf"] = pl.read_parquet(wf_file)

        return asset_data

    def _build_timeline(self, trades_df: pl.DataFrame, matrix_df: pl.DataFrame) -> pl.DataFrame:
        # Combina trades e matrix em um Log de Eventos Vertical (Event Sourcing).
        
        if trades_df is None or trades_df.is_empty():
            return pl.DataFrame()

        df_entries = trades_df.select([
            pl.col("entry_datetime").alias("datetime"),
            "trade_id",
            pl.lit(0.0).alias("pnl"),
            pl.col("lot_size"),
            pl.lit("entry").alias("event")
        ])

        df_exits = trades_df.select([
            pl.col("exit_datetime").alias("datetime"),
            "trade_id",
            pl.col("profit").alias("pnl"),
            pl.col("lot_size"),
            pl.lit("exit").alias("event")
        ])

        dfs_to_concat = [df_entries, df_exits]

        if matrix_df is not None and not matrix_df.is_empty():
            df_intra = matrix_df.select([
                pl.col("ts").alias("datetime"),
                "trade_id",
                pl.col("pnl"),
                pl.col("lot_size"),
                pl.lit("update").alias("event")
            ])
            dfs_to_concat.append(df_intra)

        timeline_df = pl.concat(dfs_to_concat, how="diagonal_relaxed")

        # Traz o ps_id do trades_df para mapear na timeline
        timeline_df = timeline_df.join(
            trades_df.select(["trade_id", "ps_id"]),
            on="trade_id",
            how="left"
        ).sort(["datetime", "trade_id"])

        return timeline_df
    

    def load_wf_prep(self, timeline_df: pl.DataFrame) -> pl.DataFrame:
        if timeline_df is None or timeline_df.is_empty():
            return None

        # Certifique-se que a coluna de tempo é temporal antes de qualquer operação
        df_updates = timeline_df.filter(pl.col("event") == "update")
        
        # Se o C++ envia string, convertemos aqui:
        df_updates = df_updates.with_columns(
            pl.col("datetime").str.to_datetime() # Converte string para Datetime real
        )

        if df_updates.is_empty():
            return None

        wide_df = (
            df_updates
            .group_by(["datetime", "ps_id"])
            .agg(pl.col("pnl").sum())
            .pivot(
                values="pnl",
                index="datetime",
                columns="ps_id"
            )
            .rename({"datetime": "ts"}) 
            .fill_null(0.0)
            .sort("ts")
        )
        
        return wide_df

    # ─────────────────────────────────────────────────────────────────────────
    # Internos
    # ─────────────────────────────────────────────────────────────────────────

    def _save_meta(self, op_path: Path, operation_name: str, meta: dict):
        meta_path = op_path / "meta"
        meta_path.mkdir(parents=True, exist_ok=True)

        full_meta = {
            "operation_name": operation_name,
            "saved_at":       datetime.now().isoformat(),
            **meta,
        }
        with open(meta_path / "operation_meta.json", "w") as f:
            json.dump(full_meta, f, indent=2, default=str)

    def _find_parquet_files(
        self,
        root:    Path,
        model:   Optional[str],
        strat:   Optional[str],
        asset:   Optional[str],
        ps_name: Optional[str],
    ) -> list:
        """Encontra arquivos parquet respeitando os filtros."""
        files = []
        for f in sorted(root.rglob("*.parquet")):
            parts = f.relative_to(root).parts  # (model, strat, asset, ps_name.parquet)
            if len(parts) < 4: continue

            m, s, a, fname = parts[0], parts[1], parts[2], parts[3]
            if model   and m != model:              continue
            if strat   and s != strat:              continue
            if asset   and a != asset:              continue
            if ps_name and f.stem != self._safe_filename(ps_name): continue

            files.append(f)
        return files

    @staticmethod
    def _safe_filename(name: str) -> str:
        """Converte ps_name para filename seguro."""
        return name.replace("/", "_").replace("\\", "_").replace(":", "_")[:200]
    

'''

    def load_trades(
        self,
        operation_name: str,
        model:   Optional[str] = None,
        strat:   Optional[str] = None,
        asset:   Optional[str] = None,
        ps_name: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Carrega trades filtrados pela nova hierarquia de pastas.
        """
        op_root = self.base_path / operation_name
        if not op_root.exists():
            print(f"   > [Storage] Operation root not found: {op_root}")
            return pl.DataFrame()

        files = []
        # O rglob vai varrer todas as subpastas chamadas "trades"
        for f in op_root.rglob("trades/*.parquet"):
            # parts -> model, strat, asset, "trades", filename
            rel_parts = f.relative_to(op_root).parts 
            if len(rel_parts) < 5: continue

            m, s, a, folder, fname = rel_parts
            
            if model and m != model: continue
            if strat and s != strat: continue
            if asset and a != asset: continue
            if ps_name and f.stem != self._safe_filename(ps_name): continue

            files.append(f)

        if not files:
            print(f"   > [Storage] No trade files matched filters.")
            return pl.DataFrame()

        dfs = [pl.read_parquet(f) for f in files]
        return pl.concat(dfs, how="diagonal_relaxed") if dfs else pl.DataFrame()


    def load_matrix_only(self, op, model, strat, asset):
        path = self.base_path / op / model / strat / asset / "matrix" / "pnl_matrix.parquet"
        if path.exists():
            return pl.read_parquet(path)
        return None

    # For local Walkforward use
    def load_pnl_matrix(
        self,
        operation_name: str,
        model:  Optional[str] = None,
        strat:  Optional[str] = None,
        asset:  Optional[str] = None,
        kind:   str = "pnl",  # "pnl" ou "lot"
    ) -> Dict[str, pl.DataFrame]:
        """
        Carrega WFM matrices de trades/.
        Retorna dict {"{model}/{strat}/{asset}": pl.DataFrame}.
        """
        trades_root = self.base_path / operation_name / "trades"
        if not trades_root.exists():
            print(f"   > [Storage] No trades found at {trades_root}")
            return {}

        filename = "pnl_matrix.parquet" if kind == "pnl" else "lot_matrix.parquet"
        results  = {}

        for f in sorted(trades_root.rglob(filename)):
            parts = f.relative_to(trades_root).parts  # (model, strat, asset, filename)
            if len(parts) < 4: continue

            m, s, a = parts[0], parts[1], parts[2]
            if model and m != model: continue
            if strat and s != strat: continue
            if asset and a != asset: continue

            key = f"{m}/{s}/{a}"
            results[key] = pl.read_parquet(f)

        return results


'''










    