from asyncio import tasks

import polars as pl, json
from pathlib import Path
from typing import Optional, Union
from datetime import datetime
from collections import OrderedDict

class Storage:
    def __init__(self, base_path: str = "Backend/results", cache_size=50):
        self.base_path = Path(base_path)
        self.cache_size = cache_size
        self._cache = OrderedDict()

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

    def save_walkforward(self, op, model, strat, asset, wf_id, config_data):
        path = self._asset_path(op, model, strat, asset) / "wfm"
        path.mkdir(parents=True, exist_ok=True)

        # Resgata o DataFrame básico gerado no analyze()
        new_df = config_data.get("oos_df")
        if new_df is None or new_df.is_empty():
            return

        file_path = path / "wf.parquet"
        
        if file_path.exists():
            old = pl.read_parquet(file_path)
            
            # Limpa o wf_id antigo do arquivo (proteção contra duplicados)
            if "wf_id" in old.columns:
                old = old.filter(pl.col("wf_id") != wf_id)
            
            new_df = pl.concat([old, new_df], how="diagonal_relaxed")
            
        if "datetime" in new_df.columns:
            new_df = new_df.sort("datetime")

        # if "ts_orig_min" in new_df.columns:
        #     new_df = new_df.drop("ts_orig_min")
            
        new_df.write_parquet(file_path)
    '''
    def save_walkforward(self, op, model, strat, asset, wf_id, config_data):
        path = self._asset_path(op, model, strat, asset) / "wfm"
        path.mkdir(parents=True, exist_ok=True)

        all_runs = config_data.get("runs", [])
        frames = []

        for run in all_runs:
            if "os_curve" in run and isinstance(run["os_curve"], pl.DataFrame):
                df = run["os_curve"].clone()

                # Drops PnL to save, use best_param and get data from trades by wf_id
                df = df.drop("pnl")

                if "ts" in df.columns:
                    df = df.rename({"ts": "datetime"})

                df = df.with_columns([
                    pl.lit(run.get("best_param", "")).alias("best_param"),
                    pl.lit(wf_id).alias("wf_id")  
                ])
                frames.append(df)

        if not frames:
            return

        new_df = pl.concat(frames)
        file_path = path / "wf.parquet"
        if file_path.exists():
            old = pl.read_parquet(file_path)
            new_df = pl.concat([old, new_df])
            
        new_df.sort("datetime").write_parquet(file_path)
    '''
    def save_operation_meta(self, op_name: str, meta_dict: dict):
        # Define o caminho da pasta da operação
        folder_path = Path(self.base_path) / op_name
        
        # CRUCIAL: Cria a pasta e todas as subpastas necessárias
        folder_path.mkdir(parents=True, exist_ok=True)
        
        path = folder_path / "operation_meta.json"
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

    # def load_meta(self, operation_name: str) -> dict:
    #     meta_file = self.base_path / operation_name / "operation_meta.json"
    #     if not meta_file.exists():
    #         return {}
    #     with open(meta_file, "r") as f:
    #         return json.load(f)

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

    # Reads trades and trades_matrix, generates a unified vertical structure with all parsets mixed in ["timeline"]
    def load(self, *args) -> dict:
        # 1. Resolver a tupla (conforme sugerido anteriormente)
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            key = args[0]
        elif len(args) == 4:
            key = args
        else:
            raise ValueError("Storage.load espera (op, m, s, a) ou 4 strings.")

        # 2. Verificar se está no cache
        if key in self._cache:
            # Move para o final para marcar como "recentemente usado"
            self._cache.move_to_end(key)
            return self._cache[key]

        # 3. Se não estiver, carregar do disco (Sua lógica original)
        asset_data = self._execute_load(*key)

        # 4. Adicionar ao cache e controlar tamanho
        self._cache[key] = asset_data
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False) # Remove o mais antigo (primeiro)

        return asset_data
    
    def _execute_load(self, op, model, strat, asset):
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

        if not asset_data:
            print(f"    < [Storage._execute_load] ⚠️ Warning: No data found for {op}/{model}/{strat}/{asset}")

        return asset_data



    # || Operation.py Use || # 

    def _load_matrix_only(self, op, model, strat, asset):
        path = self.base_path / op / model / strat / asset / "matrix" / "pnl_matrix.parquet"
        if path.exists():
            return pl.read_parquet(path)
        return None
    def load_walkforward_matrix(self, 
                                key, 
                                side_val: str="BOTH", 
                                wf_ids: Optional[Union[str, list]] = None, 
                                start_dt: str=None, 
                                end_dt: str=None) -> Optional[pl.DataFrame]:              
        # Implemented asynchronous parallel processing via pl.collect_all on a list of Lazy execution plans, 
        # allowing the engine to compute multiple walkforward curves simultaneously across all CPU cores 
        # while minimizing memory copies.

        # 1. Carrega as estruturas do disco
        asset_data = self.load(key)
        trades_df = asset_data.get("timeline")
        wf_map = asset_data.get("wf")
        if trades_df is None or wf_map is None: 
            print(f"    < [Storage.load_walkforward_matrix] trades_df or wf_map None")
            return None

        try:
            # 2. Normalização do mapa do Walkforward
            if "best_param" in wf_map.columns:
                wf_map = wf_map.rename({"best_param": "ps_id"})

            if start_dt:
                wf_map = wf_map.filter(pl.col("datetime") >= start_dt)
            if end_dt:
                wf_map = wf_map.filter(pl.col("datetime") <= end_dt)
            
            if wf_ids is not None:
                search_ids = [str(wf_ids)] if isinstance(wf_ids, str) else [str(i) for i in wf_ids]
                wf_map = wf_map.filter(pl.col("wf_id").cast(pl.Utf8).is_in(search_ids))

            # 3. Mapeamento Reverso (Resgata qual Parameter Set gerou aquele trade_id)
            # Como a timeline original salvou trade_id na coluna ps_id, precisamos da matriz real
            # para correlacionar qual coluna de parâmetro gerou a operação.
            op, model, strat, asset = key.split("/")
            pnl_matrix = self._load_pnl_matrix_only(op, model, strat, asset)
            
            if pnl_matrix is None:
                print(f"    < [Storage.load_walkforward_matrix] pnl_matrix não encontrada para remapeamento.")
                return None

            # Cria um dicionário de mapeamento mapeando trade_id -> real_parameter_set_id
            # Evita joins pesados criando um mapa estático na memória via Polars (extremamente rápido)
            if "trade_id" in pnl_matrix.columns:
                # Se a matriz bruta guardou a relação vertical mapeada
                map_df = pnl_matrix.select(["trade_id", "ps_id"]).drop_nulls().unique()
                trade_to_ps_dict = dict(zip(map_df["trade_id"].to_list(), map_df["ps_id"].to_list()))
            else:
                # Backup de segurança: Tenta buscar as colunas de parsets disponíveis na matriz
                ps_ids_columns = [c for c in pnl_matrix.columns if c not in {"ts", "ts_orig_min", "ts_orig_max"}]
                # Se os nomes das colunas já forem os parsets reais, o seu active_ps buscará por eles.
                trade_to_ps_dict = {}

            # 4. Preparação Lazy do histórico de trades
            trades_lazy = trades_df.lazy().select(["datetime", "ps_id", "pnl", "lot_size"])
            
            if start_dt:
                trades_lazy = trades_lazy.filter(pl.col("datetime") >= start_dt)
            if end_dt:
                trades_lazy = trades_lazy.filter(pl.col("datetime") <= end_dt)

            # Filtros por direção (Long/Short)
            if side_val.upper() != "BOTH":
                s_filter = side_val.lower()
                if s_filter == "long":
                    trades_lazy = trades_lazy.filter(pl.col("lot_size") > 0)
                elif s_filter == "short":
                    trades_lazy = trades_lazy.filter(pl.col("lot_size") < 0)

            # Recorta colunas e aplica o remapeamento usando replace do Polars para converter trade_id em Parameter Set
            trades_lazy = trades_lazy.select([
                "datetime", 
                # 🔥 CORREÇÃO CRÍTICA: Transforma a coluna mascarada com trade_id no id do parâmetro real correspondente
                pl.col("ps_id").replace(trade_to_ps_dict, default=pl.col("ps_id")).alias("real_parameter_id"),
                "pnl"
            ])

            unique_wfs = wf_map.get_column("wf_id").unique().to_list()

            # 5. Geração das tarefas paralelas em formato Lazy
            tasks = []
            for wid in unique_wfs:
                wf_map_lazy = wf_map.filter(pl.col("wf_id") == wid).select([
                    pl.col("datetime").alias("map_dt"),
                    pl.col("ps_id").alias("active_ps")
                ]).lazy().sort("map_dt")

                # O join_asof agora casa os tempos e o filtro finalmente valida dados reais
                plan = trades_lazy.join_asof(
                    wf_map_lazy,
                    left_on="datetime",
                    right_on="map_dt",
                    strategy="backward"
                ).filter(
                    pl.col("real_parameter_id") == pl.col("active_ps")  # 🔥 Agora a comparação bate com sucesso!
                ).select([
                    "datetime",
                    "pnl",
                    pl.lit(str(wid)).alias("wf_id")
                ])

                tasks.append(plan)

            if not tasks: 
                print(f"    < [Storage.load_walkforward_matrix] No tasks")
                return None

            # 6. Execução nativa paralela em C++ do Polars (usa todos os cores)
            results = pl.collect_all(tasks)
            if not results or all(r.is_empty() for r in results): 
                print(f"    < [Storage.load_walkforward_matrix] No results or empty dataframes computed")
                return None

            # 7. Concatena e pivota os resultados gerando a matriz final
            return pl.concat(results).pivot(
                index="datetime",
                on="wf_id",
                values="pnl",
                aggregate_function="sum"
            ).fill_null(0.0).sort("datetime")

        except Exception as e:
            print(f"    < [Storage.] Error: {e}")
            return None

    # Retorna um DataFrame com colunas datetime e wf_id com res_price de cada walkforward 
    def load_walkforward_matrix_v2(self,
                                   key, # tuple (op, model, strat, asset) or 4 individual strings
                                   res_price: str="perc",
                                   side_val: str="BOTH",
                                   wf_ids: Optional[Union[str, list]] = None,
                                   timeline_df: Optional[pl.DataFrame] = None, # Permite passar a timeline já processada para evitar recálculos
                                   wf_map: Optional[pl.DataFrame] = None, # Permite passar o mapa do walkforward já processado para evitar recálculos
                                   start_dt: str=None,
                                   end_dt: str=None) -> Optional[pl.DataFrame]:
        
        # Loads and validates data
        if timeline_df is None or wf_map is None:
            asset_data = self.load(key)
            if timeline_df is None:
                timeline_df = asset_data.get("timeline")
            if wf_map is None:
                wf_map = asset_data.get("wf")

        if timeline_df is None or wf_map is None: 
            print(f"    < [Storage.load_walkforward_matrix] timeline_df or wf_map None")
            return None
        
        #try:
        # Normalizes Walkforward parameter mapping columns
        if "best_param" in wf_map.columns and "ps_id" not in wf_map.columns:
            wf_map = wf_map.rename({"best_param": "ps_id"})

        # Applies cronological filters
        if start_dt:
            wf_map = wf_map.filter(pl.col("datetime") >= start_dt)
        if end_dt:
            wf_map = wf_map.filter(pl.col("datetime") <= end_dt)
        
        # Optional wf_id filtering
        if wf_ids is not None:
            search_ids = [str(wf_ids)] if isinstance(wf_ids, str) else [str(i) for i in wf_ids]
            wf_map = wf_map.filter(pl.col("wf_id").cast(pl.Utf8).is_in(search_ids))

        # Lazy historical and temporal preparation and optimization
        timeline_lazy = timeline_df.lazy()
        
        if start_dt:
            timeline_lazy = timeline_lazy.filter(pl.col("datetime") >= start_dt)
        if end_dt:
            timeline_lazy = timeline_lazy.filter(pl.col("datetime") <= end_dt)

        # Optional filters by trade direction
        if side_val.upper() != "BOTH":
            s_filter = side_val.upper()
            lot_col_name = "lot_size" if "lot_size" in timeline_df.columns else "lot"

            if lot_col_name in timeline_df.columns:
                if s_filter == "LONG":
                    timeline_lazy = timeline_lazy.filter(pl.col(lot_col_name) > 0)
                elif s_filter == "SHORT":
                    timeline_lazy = timeline_lazy.filter(pl.col(lot_col_name) < 0)

        # Identifies which walkforward windows/config needs to be processed
        unique_wfs = wf_map.get_column("wf_id").unique().to_list()

        # Parallel task generation in lazy format, one per walkforward ID
        tasks = []
        for wid in unique_wfs:
            wf_map_lazy = wf_map.filter(pl.col("wf_id") == wid).select([
                pl.col("datetime").alias("map_dt"),
                pl.col("ps_id").alias("active_ps")
            ]).lazy().sort("map_dt")

            # Retroactive join_asof
            plan = timeline_lazy.join_asof(
                wf_map_lazy,
                left_on="datetime",
                right_on="map_dt",
                strategy="backward"
            ).filter(
                pl.col("ps_id") == pl.col("active_ps")  
            ).select([
                "datetime",
                pl.col(res_price).alias(res_price),  
                pl.lit(str(wid)).alias("wf_id")
            ])
            tasks.append(plan)
            
        if not tasks:
            print(f"    < [Storage.load_walkforward_matrix] No tasks generated.")
            return None
        
        # Multi-threaded execution with polars
        results = pl.collect_all(tasks)
        if not results or all(r.is_empty() for r in results):
            print(f"    < [Storage.load_walkforward_matrix] No trade corresponded with OOS mapping.")
            return None
        
        # Vertical concatenation and pivoting to generate final matrix
        return pl.concat(results).pivot(
            index="datetime",
            on="wf_id",
            values=res_price,
            aggregate_function="sum"
        ).fill_null(0.0).sort("datetime")
    
        #except Exception as e:
        #    print(f"    < [Storage.load_walkforward_matrix] Critical error in construction: {e}")
        #    return None
    
    
    # For Operation.py use
    def load_wf_prep(self, timeline_df: pl.DataFrame, price: str = "pnl", events_to_include: list = ["entry", "exit", "update"]) -> pl.DataFrame:
        if timeline_df is None or timeline_df.is_empty():
            print(f"   < [Storage.load_wf_prep] timeline_df is None or empty")
            return None

        return (
            timeline_df
            .filter(pl.col("event").is_in(events_to_include))
            .group_by(["datetime", "ps_id"])
            .agg(pl.col(price).sum())
            .pivot(values=price, index="datetime", columns="ps_id")
            .rename({"datetime": "ts"}) 
            .fill_null(0.0)
            .sort("ts")
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Internos
    # ─────────────────────────────────────────────────────────────────────────

    def _build_timeline(self, 
                        trades_df: pl.DataFrame, 
                        matrix_df: pl.DataFrame, 
                        fmt: str= "%Y%m%d%H%M%S") -> pl.DataFrame:
        
        if trades_df is None or trades_df.is_empty():
            return pl.DataFrame()
        
        # Formato numérico exato vindo do C++ (ex: 20210412013000)
        def cast_datetime(column_name, df_source, alias="datetime"):
            col = pl.col(column_name).replace(0, None)
            
            # Se o tipo da coluna no schema já for Datetime/Date, apenas renomeia
            if df_source.schema.get(column_name) in [pl.Datetime, pl.Date]:
                return col.alias(alias)
                
            # Caso contrário, tenta string formatada do C++ e dá fallback para Epoch/Int
            return pl.coalesce([
                col.cast(pl.Utf8).str.to_datetime(fmt, strict=False),
                col.cast(pl.Int64).cast(pl.Datetime("us"))
            ]).alias(alias)
        
        # Gets original relation between trade_id and ps_id
        trade_to_ps = trades_df.select(["trade_id", "ps_id", "asset"]).unique()

        # 1. ENTRIES
        entries_df = trades_df.select([
            cast_datetime("entry_datetime", trades_df),
            pl.col("asset"),
            pl.col("trade_id").cast(pl.Utf8),
            pl.col("ps_id").cast(pl.Utf8),
            pl.lit(0.0).alias('pnl'),
            pl.lit(0.0).alias('perc'),
            pl.col("lot_size"),
            pl.col('margin_required'),
            pl.lit(0.0).alias('mae'),
            pl.lit(0.0).alias('mfe'),
            pl.lit("entry").alias("event")
        ])
            
        # 2. EXITS
        exits_df = trades_df.select([
            cast_datetime("exit_datetime", trades_df),
            pl.col("asset"),
            pl.col("trade_id").cast(pl.Utf8), 
            pl.col("ps_id").cast(pl.Utf8),
            pl.col("pnl"),
            pl.col('perc'),
            pl.col("exit_lot_size").alias("lot_size"),
            pl.col('exit_margin').alias("margin_required"),
            pl.col('mae'),
            pl.col('mfe'),
            pl.lit("exit").alias("event")
        ])

        dfs_to_concat = [entries_df, exits_df]

        # 3. Trade Updates (MATRIX)
        if matrix_df is not None and not matrix_df.is_empty():
            updates_df = matrix_df.join(trade_to_ps, on="trade_id", how="left").select([
                cast_datetime("ts", matrix_df),
                pl.col("asset"),
                pl.col("trade_id").cast(pl.Utf8),
                pl.col("ps_id").cast(pl.Utf8),
                pl.col("pnl"),
                pl.col("perc"),
                pl.col("lot_size"),
                pl.col("margin_required"),
                pl.col("mae"),
                pl.col("mfe"),
                pl.lit("update").alias("event")
            ])
            dfs_to_concat.append(updates_df)
        timeline_df = pl.concat(dfs_to_concat).sort("datetime")

        return timeline_df
    
    """
        def _build_timeline(self, trades_df: pl.DataFrame, matrix_df: pl.DataFrame, fmt: str= "%Y%m%d%H%M%S") -> pl.DataFrame:
        if trades_df is None or trades_df.is_empty():
            return pl.DataFrame()
        
        # Formato numérico exato vindo do C++ (ex: 20210412013000)
        def cast_datetime(column_name, df_source, alias="datetime"):
            col = pl.col(column_name).replace(0, None)
            
            # Se o tipo da coluna no schema já for Datetime/Date, apenas renomeia
            if df_source.schema.get(column_name) in [pl.Datetime, pl.Date]:
                return col.alias(alias)
                
            # Caso contrário, tenta string formatada do C++ e dá fallback para Epoch/Int
            return pl.coalesce([
                col.cast(pl.Utf8).str.to_datetime(fmt, strict=False),
                col.cast(pl.Int64).cast(pl.Datetime("us"))
            ]).alias(alias)

        # 1. ENTRIES
        entries_df = trades_df.select([
            cast_datetime("entry_datetime", trades_df),
            pl.col("trade_id").cast(pl.Utf8).alias("ps_id"),
            pl.lit(0.0).alias('pnl'),
            pl.lit(0.0).alias('perc'),
            pl.col("lot_size"),
            pl.lit(0.0).alias('margin_required'),
            pl.lit(0.0).alias('mae'),
            pl.lit(0.0).alias('mfe'),
            pl.lit("entry").alias("event")
        ])
            
        # 2. EXITS
        exits_df = trades_df.select([
            cast_datetime("exit_datetime", trades_df),
            pl.col("trade_id").cast(pl.Utf8).alias("ps_id"), # trade_id
            pl.col("pnl"),
            pl.col('perc'),
            pl.col("lot_size"),
            pl.col('margin_required'),
            pl.col('mae'),
            pl.col('mfe'),
            pl.lit("exit").alias("event")
        ])

        dfs_to_concat = [entries_df, exits_df]

        # 3. Trade Updates (MATRIX)
        if matrix_df is not None and not matrix_df.is_empty():
            updates_df = matrix_df.select([
                cast_datetime("ts", matrix_df),
                pl.col("trade_id").cast(pl.Utf8).alias("ps_id"),
                pl.col("pnl"),
                pl.col("perc"),
                pl.col("lot_size"),
                pl.col("margin_required"),
                pl.col("mae"),
                pl.col("mfe"),
                pl.lit("update").alias("event")
            ])
            dfs_to_concat.append(updates_df)
        results = pl.concat(dfs_to_concat).sort("datetime")

        return results
    """



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










    