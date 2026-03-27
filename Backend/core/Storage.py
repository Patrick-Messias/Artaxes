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

    def _get_asset_path(self, op: str, model: str, strat: str, asset: str) -> Path:
        return self.base_path / op / model / strat / asset
    
    def save_matrix_data(self, op: str, model: str, strat: str, asset: str,
                         pnl_df: Optional[pl.DataFrame], lot_df: Optional[pl.DataFrame]):
        path = self._get_asset_path(op, model, strat, asset) / "matrix"
        path.mkdir(parents=True, exist_ok=True)

        if pnl_df is not None: 
            pnl_df.write_parquet(path / "pnl_matrix.parquet")
        if lot_df is not None:
            lot_df.write_parquet(path / "lot_matrix.parquet")

    def save_walkforward(self, op: str, model: str, strat: str, asset: str, 
                        wf_id: str, all_runs: list):
        # Salva cada configuração de WFM como um Parquet independente.
        # Ex: wf_id = "12_4_4" -> wf_12_4_4.parquet
        
        path = self._get_asset_path(op, model, strat, asset) / "wfm"
        path.mkdir(parents=True, exist_ok=True)

        oos_frames = []
        for run in all_runs:
            # Pega o DataFrame do Walkforward
            if "os_curve" in run and isinstance(run["os_curve"], pl.DataFrame):
                os_df = run["os_curve"].clone()
                
                # Renomeia "ts" para "datetime" para o Portfolio Simulator alinhar depois
                if "ts" in os_df.columns:
                    os_df = os_df.rename({"ts": "datetime"})
                
                # Injeta qual foi o param set escolhido nesta janela específica
                best_param = run.get("best_param", "")
                os_df = os_df.with_columns(pl.lit(best_param).alias("best_param"))
                
                oos_frames.append(os_df)
        
        if oos_frames:
            filename = f"wf_{wf_id}.parquet"
            # Concatena as janelas OOS e salva
            full_oos_curve = pl.concat(oos_frames).sort("datetime")
            full_oos_curve.write_parquet(path / filename)
            #print(f"      > [Storage] Saved WFM: {filename}")


    def save_individual_trade(self, op, model, strat, asset, ps_name, trades_df):
        path = self.base_path / op / model / strat / asset / "trades"
        path.mkdir(parents=True, exist_ok=True)
        
        # Sanitiza o nome do arquivo (remove caracteres que o Windows não gosta)
        safe_ps_name = "".join([c if c.isalnum() or c in "-_" else "_" for c in ps_name])
        trades_df.write_parquet(path / f"{safe_ps_name}.parquet")

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

    def load_meta(self, operation_name: str) -> dict:
        meta_file = self.base_path / operation_name / "meta" / "operation_meta.json"
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

    def load(self, op: str, model: str, strat: str, asset: str) -> Dict[str, Any]:
        asset_path = self._get_asset_path(op, model, strat, asset)
        asset_data = {}

        # --- LOAD MATRIX (PNL + LOT) ---
        matrix_path = asset_path / "matrix"
        if matrix_path.exists():
            pnl_file = matrix_path / "pnl_matrix.parquet"
            lot_file = matrix_path / "lot_matrix.parquet"
            
            if pnl_file.exists():
                df_pnl = pl.read_parquet(pnl_file)
                df_lot = pl.read_parquet(lot_file) if lot_file.exists() else None
                time_col = df_pnl.columns[0]
                
                for ps in [c for c in df_pnl.columns if c != time_col]:
                    temp = df_pnl.select([pl.col(time_col), pl.col(ps).alias("pnl")])
                    if df_lot is not None and ps in df_lot.columns:
                        temp = temp.join(df_lot.select([time_col, ps]), on=time_col).rename({ps: "lot"})
                    else:
                        temp = temp.with_columns(pl.lit(0.0).alias("lot"))
                    asset_data[ps] = temp

        # --- LOAD WFM (Individual Parquets) ---
        wfm_path = asset_path / "wfm"
        if wfm_path.exists():
            for wf_file in wfm_path.glob("*.parquet"):
                asset_data[f"wf_{wf_file.stem}"] = pl.read_parquet(wf_file)

        return asset_data

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
    












    