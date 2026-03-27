import polars as pl, json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

class Storage:
    """
    Gerencia persistência dos resultados da Operation em Parquet + JSON metadata.

    Estrutura de diretórios:
        {base_path}/{operation_name}/
            meta/
                operation_meta.json     → metadata da operation (datas, assets, params)
            trades/
                {model}/{strat}/{asset}/{ps_name}.parquet
            wfm/
                {model}/{strat}/{asset}/pnl_matrix.parquet
                {model}/{strat}/{asset}/lot_matrix.parquet
    """

    def __init__(self, base_path: str = "Backend/results"):
        self.base_path = Path(base_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, operation_name: str, results_map: dict, meta: dict = None):
        """
        Salva todos os resultados de uma Operation.
        Chamado por _save_and_clean().
        """
        op_path = self.base_path / operation_name
        op_path.mkdir(parents=True, exist_ok=True)
 
        self._save_meta(op_path, operation_name, meta or {})
        n_trades, n_wfm = 0, 0
 
        models = results_map.get(operation_name, {}).get("models", {})
        for model_name, model_data in models.items():
            for strat_name, strat_data in model_data.get("strats", {}).items():
                for asset_name, asset_data in strat_data.get("assets", {}).items():
 
                    trades_path = op_path / "trades" / model_name / strat_name / asset_name
                    trades_path.mkdir(parents=True, exist_ok=True)
 
                    # ── Trades individuais ────────────────────────────────────
                    for ps_name, ps_data in asset_data.get("param_sets", {}).items():
                        trades = ps_data.get("trades")
                        if trades is None: continue
                        if isinstance(trades, pl.DataFrame) and trades.is_empty(): continue
                        if isinstance(trades, list) and len(trades) == 0: continue
 
                        df        = trades if isinstance(trades, pl.DataFrame) else pl.DataFrame(trades)
                        safe_name = self._safe_filename(ps_name)
                        df.write_parquet(trades_path / f"{safe_name}.parquet")
                        n_trades += 1
 
                    # ── WFM matrices (pnl + lot) ──────────────────────────────
                    pnl_matrix = asset_data.get("wfm_matrix_data")
                    if pnl_matrix is not None and not pnl_matrix.is_empty():
                        pnl_matrix.write_parquet(trades_path / "pnl_matrix.parquet")
                        n_wfm += 1
 
                    lot_matrix = asset_data.get("wfm_lot_data")
                    if lot_matrix is not None and not lot_matrix.is_empty():
                        lot_matrix.write_parquet(trades_path / "lot_matrix.parquet")
 
        print(f"   > [Storage] Saved {n_trades} trade files and {n_wfm} WFM matrices")
        print(f"   > [Storage] Path: {op_path.resolve()}")

    # ─────────────────────────────────────────────────────────────────────────
    # Load
    # ─────────────────────────────────────────────────────────────────────────

    def load_trades(
        self,
        operation_name: str,
        model:   Optional[str] = None,
        strat:   Optional[str] = None,
        asset:   Optional[str] = None,
        ps_name: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Carrega trades filtrados por model/strat/asset/ps_name.
        Se algum filtro for None, carrega todos os níveis abaixo.
        Retorna DataFrame único concatenado.
        """
        trades_root = self.base_path / operation_name / "trades"
        if not trades_root.exists():
            print(f"   > [Storage] No trades found at {trades_root}")
            return pl.DataFrame()

        files = self._find_parquet_files(trades_root, model, strat, asset, ps_name)
        if not files:
            print(f"   > [Storage] No trade files matched filters.")
            return pl.DataFrame()

        dfs = []
        for f in files:
            if f.stem in ("pnl_matrix", "lot_matrix", "walkforward_result"):
                continue
            df = pl.read_parquet(f)
            dfs.append(df)

        return pl.concat(dfs, how="diagonal_relaxed") if dfs else pl.DataFrame()

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

    def load(self, op_name: str, model: str, strat: str, asset: str, 
            pnl_saved_name="pnl_matrix.parquet",
            lot_saved_name="lot_matrix.parquet",
            wf_saved_name="all_wf_results.json"
            ) -> Dict[str, pl.DataFrame]:
        folder_path = self.base_path / op_name / "trades" / model / strat / asset
        asset_container = {}
       
        # Creates paths and loads data
        pnl_path = folder_path / pnl_saved_name
        lot_path = folder_path / lot_saved_name
        wf_path = folder_path / wf_saved_name

        if not pnl_path.exists(): 
            print(f"< [Storage] path: {pnl_path} doesn't exist")
            return {}

        df_pnl = pl.read_parquet(pnl_path)
        time_col = df_pnl.columns[0]
        parsets = [c for c in df_pnl.columns if c != time_col]
  
        # Tries loading lots to cross the data
        df_lot = pl.read_parquet(lot_path) if lot_path.exists() else None

        # Creates one DataFrame per Parset: [datetime, pnl, lot]
        for ps in parsets:
            # Selects datetime and the specific columns fo the parset
            ps_df = df_pnl.select([pl.col(time_col), pl.col(ps).alias("pnl")])
            
            # If there lot matrix, joins
            if df_lot is not None and ps in df_lot.columns:
                lot_series = df_lot.select([pl.col(time_col), pl.col(ps).alias("lot")])
                ps_df = ps_df.join(lot_series, on=time_col, how="left")
            else:
                ps_df = ps_df.with_columns(pl.lit(0.0).alias("lot"))

            asset_container[ps] = ps_df

        # Loads Walkorward as DataFrames
        if wf_path.exists():
            with open(wf_path, 'r') as f:
                wf_data = json.load(f)
                for wf_id, wf_content in wf_data.items():
                    # Transforms every walkforward's windows in a DataFrame for easier consult
                    asset_container[f"wf_{wf_id}"] = pl.DataFrame(wf_content)

        return asset_container

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


    def _save_walkforward(self, wfm_path, all_wf_results: dict, filename: str = "all_wf_results.json"):
        try:
            # Remove is_df (pl.DataFrame pesado, reconstruído na próxima rodada)
            import copy
            results_to_save = copy.deepcopy(all_wf_results)
            for cfg_data in results_to_save.values():
                if not isinstance(cfg_data, dict): continue
                for run in cfg_data.get("runs", []):
                    run.pop("is_df", None)

            def _serialize(obj):
                if isinstance(obj, pl.DataFrame):  return obj.to_dicts()
                if isinstance(obj, pl.Series):     return obj.to_list()
                if hasattr(obj, 'isoformat'):       return obj.isoformat()
                if hasattr(obj, 'tolist'):          return obj.tolist()
                if hasattr(obj, 'item'):            return obj.item()
                return str(obj)

            with open(wfm_path / filename, "w") as f:
                json.dump(results_to_save, f, indent=2, default=_serialize)
        except Exception as e:
            print(f"   > [Storage] Warning: could not save walkforward results: {e}")


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
    












    