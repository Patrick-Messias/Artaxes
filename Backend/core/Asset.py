"""
Asset.py
========
Arquitetura:
    DataSource (ABC)          — interface comum para todas as fontes
        LocalSource           — lê parquet de Backend/Assets/
        FileSource            — importa CSV / XLSX
        YFinanceSource        — baixa do yfinance  (stub — implementar depois)
        FredSource            — baixa do FRED      (stub — implementar depois)

    AssetRegistry             — lê/escreve registry.json por market
    Asset                     — instância de um ativo; delega load() para DataSource
    AssetPortfolio            — coleção de Assets
"""

import polars as pl
import json, os, re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union


# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

ASSETS_BASE = Path("Backend/Assets")   # raiz dos parquets e registries
ASSETS_BASE = Path(__file__).parent.parent / "Assets"

SourceType = Literal["local", "file", "yfinance", "fred"]

# ═════════════════════════════════════════════════════════════════════════════
# DATA SOURCES  (Strategy Pattern)
# ═════════════════════════════════════════════════════════════════════════════

class DataSource(ABC):
    # Interface comum — toda source deve implementar load().

    @abstractmethod
    def load(
        self,
        name:             str,
        timeframe:        str,
        date_start:       Optional[str],
        date_end:         Optional[str],
        **kwargs,
    ) -> pl.DataFrame:
        ...

    # ── Helpers compartilhados ────────────────────────────────────────────────

    @staticmethod
    def _filter_dates(df: pl.DataFrame, date_start: Optional[str], date_end: Optional[str]) -> pl.DataFrame:
        if date_start:
            df = df.filter(pl.col("datetime") >= pl.lit(date_start).str.to_datetime("%Y-%m-%d"))
        if date_end:
            df = df.filter(pl.col("datetime") <= pl.lit(date_end).str.to_datetime("%Y-%m-%d"))
        return df

    @staticmethod
    def _normalize(df: pl.DataFrame, datetime_col: str = "datetime", datetime_format: str = None) -> pl.DataFrame:
        df = df.rename({c: c.lower().strip() for c in df.columns})

        # Candidatos em ordem de prioridade
        candidates = [c for c in [datetime_col, "datetime", "time", "date", "timestamp"] if c]
        dt_col = next((c for c in candidates if c in df.columns), None)

        if dt_col is None:
            if "date" in df.columns and "time" in df.columns:
                df = df.with_columns(
                    (pl.col("date").cast(pl.Utf8) + pl.lit(" ") + pl.col("time").cast(pl.Utf8))
                    .alias("datetime")
                )
                dt_col = "datetime"
            else:
                raise ValueError(f"No datetime column found. Columns: {df.columns}")

        if dt_col != "datetime":
            df = df.rename({dt_col: "datetime"})

        if df["datetime"].dtype not in (pl.Datetime, pl.Date):
            df = df.with_columns(
                pl.col("datetime")
                .str.replace_all(r"\.", "-")
                .str.to_datetime(format=datetime_format, strict=False)
            )

        if df["datetime"].dtype == pl.Date:
            df = df.with_columns(pl.col("datetime").cast(pl.Datetime))

        return df.sort("datetime").unique(subset=["datetime"], keep="first")

# ── Local ─────────────────────────────────────────────────────────────────────

class LocalSource(DataSource):
    # Lê parquet de Backend/Assets/{market}/{name}_{timeframe}.parquet

    def load(self, name, timeframe, date_start=None, date_end=None, **kwargs) -> pl.DataFrame:
        market = kwargs.get("market", "")
        path   = ASSETS_BASE / market / f"{name}_{timeframe}.parquet"

        if not path.exists():
            raise FileNotFoundError(f"[LocalSource] Parquet not found: {path}")

        df = pl.read_parquet(path)
        return self._filter_dates(df, date_start, date_end)

# ── File (CSV / XLSX) ─────────────────────────────────────────────────────────

class FileSource(DataSource):
    # Importa CSV ou XLSX, salva como parquet e atualiza o registry.
    # Parâmetros extras via kwargs:
    #     path             : caminho do arquivo (obrigatório)
    #     datetime_col     : nome da coluna datetime no arquivo
    #     datetime_format  : formato strptime se a inferência falhar
    #     delimiter        : separador CSV (padrão ',')
    #     save             : bool — salva parquet após importar (padrão True)
    #     update_reason    : str — motivo para o registry
    

    def load(self, name, timeframe, date_start=None, date_end=None, **kwargs) -> pl.DataFrame:
        file_path = kwargs.get("path")
        if not file_path:
            raise ValueError("[FileSource] 'path' kwarg is required")

        source    = Path(file_path)
        ext       = source.suffix.lower()
        delimiter = kwargs.get("delimiter", ",")

        if ext == ".csv":
            df = pl.read_csv(source, separator=delimiter, try_parse_dates=False)
        elif ext in (".xlsx", ".xls"):
            df = pl.read_excel(source)
        else:
            raise ValueError(f"[FileSource] Unsupported format: {ext}")

        df = self._normalize(
            df,
            datetime_col=kwargs.get("datetime_col") or "datetime",
            datetime_format=kwargs.get("datetime_format"),
        )

        # Salva parquet + atualiza registry
        if kwargs.get("save", True):
            market  = kwargs.get("market", "")
            out_dir = ASSETS_BASE / market
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{name}_{timeframe}.parquet"
            df.write_parquet(out_file)
            print(f"   > [FileSource] Saved → {out_file}")

        return self._filter_dates(df, date_start, date_end)

# ── YFinance (stub) ───────────────────────────────────────────────────────────

class YFinanceSource(DataSource):
    # Baixa dados do Yahoo Finance. Requer: pip install yfinance

    def load(self, name, timeframe, date_start=None, date_end=None, **kwargs) -> pl.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("[YFinanceSource] Install yfinance: pip install yfinance")

        # Mapeia timeframe interno → intervalo yfinance
        _tf_map = {
            "M1": "1m",  "M5": "5m",  "M15": "15m", "M30": "30m",
            "H1": "1h",  "H4": "4h",  "D1": "1d",   "W1": "1wk",
        }
        interval = _tf_map.get(timeframe.upper(), "1d")

        ticker = yf.Ticker(name)
        raw    = ticker.history(
            start=date_start,
            end=date_end,
            interval=interval,
            auto_adjust=True,
        )

        if raw.empty:
            raise ValueError(f"[YFinanceSource] No data returned for {name}")

        raw = raw.reset_index()
        raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
        df = pl.from_pandas(raw)

        # Garante coluna datetime
        dt_col = next((c for c in df.columns if "date" in c or "time" in c), None)
        if dt_col and dt_col != "datetime":
            df = df.rename({dt_col: "datetime"})
        if df["datetime"].dtype == pl.Date:
            df = df.with_columns(pl.col("datetime").cast(pl.Datetime))

        df = df.sort("datetime")

        if kwargs.get("save", True):
            market  = kwargs.get("market", "")
            out_dir = ASSETS_BASE / market
            out_dir.mkdir(parents=True, exist_ok=True)
            df.write_parquet(out_dir / f"{name}_{timeframe}.parquet")

        return df

# ── FRED (stub) ───────────────────────────────────────────────────────────────

class FredSource(DataSource):
    # Baixa séries do FRED. Requer: pip install fredapi

    def load(self, name, timeframe, date_start=None, date_end=None, **kwargs) -> pl.DataFrame:
        # TODO: implementar
        # series_id = kwargs.get("series_id", name)
        # fred = Fred(api_key=kwargs.get("api_key"))
        raise NotImplementedError("[FredSource] Not yet implemented")

# ── Registry de sources ───────────────────────────────────────────────────────

_SOURCE_REGISTRY: Dict[str, DataSource] = {
    "local":    LocalSource(),
    "file":     FileSource(),
    "yfinance": YFinanceSource(),
    "fred":     FredSource(),
}

# ═════════════════════════════════════════════════════════════════════════════
# ASSET REGISTRY  (metadata por market)
# ═════════════════════════════════════════════════════════════════════════════

class AssetRegistry:    
    # Lê e escreve Backend/Assets/{market}/registry.json

    # Estrutura:
    # {
    #   "market": "forex",
    #   "asset_type": "currency_pair",
    #   "assets": {
    #     "EURUSD": {
    #       "asset_params": { "tick": 0.0001, ... },
    #       "timeframes": {
    #         "M15": {
    #           "date_start": "...", "date_end": "...", "rows": 38420,
    #           "last_updated": "...", "last_update_reason": "..."
    #         }
    #       }
    #     }
    #   }
    # }
    

    def __init__(self, market: str, asset_type: str = ""):
        self.market     = market
        self.asset_type = asset_type
        self.path       = ASSETS_BASE / market / "registry.json"
        self._data      = self._load()

    # ── IO ────────────────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path, "r") as f:
                return json.load(f)
        return {"market": self.market, "asset_type": self.asset_type, "assets": {}}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=4, default=str)

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_asset(self, name: str) -> Optional[dict]:
        return self._data["assets"].get(name)

    def get_timeframe(self, name: str, timeframe: str) -> Optional[dict]:
        asset = self.get_asset(name)
        if asset:
            return asset.get("timeframes", {}).get(timeframe)
        return None

    def list_assets(self) -> List[str]:
        return list(self._data["assets"].keys())

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert(
        self,
        name:          str,
        timeframe:     str,
        df:            pl.DataFrame,
        asset_params:  dict,
        update_reason: str = "manual update",
    ):
        """Cria ou atualiza entrada de um ativo/timeframe no registry."""
        assets = self._data.setdefault("assets", {})
        entry  = assets.setdefault(name, {"asset_params": asset_params, "timeframes": {}})

        # Atualiza asset_params se passado
        entry["asset_params"] = asset_params

        entry["timeframes"][timeframe] = {
            "date_start":         str(df["datetime"].min()),
            "date_end":           str(df["datetime"].max()),
            "rows":               len(df),
            "last_updated":       datetime.now().isoformat(timespec="seconds"),
            "last_update_reason": update_reason,
        }

        self.save()

    # ── Print ─────────────────────────────────────────────────────────────────

    def summary(self):
        print(f"\n{'═'*60}")
        print(f"  Registry — {self.market.upper()}")
        print(f"{'═'*60}")
        for name, data in self._data.get("assets", {}).items():
            print(f"\n  {name}")
            for tf, meta in data.get("timeframes", {}).items():
                print(f"    {tf:6s}  {meta['date_start']} → {meta['date_end']}  "
                      f"({meta['rows']:,} rows)  updated: {meta['last_updated']}")
        print(f"\n{'═'*60}\n")

# ═════════════════════════════════════════════════════════════════════════════
# ASSET
# ═════════════════════════════════════════════════════════════════════════════

class Asset:    
    # Representa um ativo financeiro.

    # Responsabilidades:
    #     - Armazena parâmetros de mercado (tick, lot, leverage, etc.)
    #     - Delega carregamento de dados para DataSource via load()
    #     - Delega metadados para AssetRegistry via convert()
    #     - Expõe data_get() para compatibilidade com Operation/Walkforward
    

    ASSET_PARAMS: Dict = {
        "futures": {
            "b3": {
                "WIN$": {
                    "tick": 1, "tick_fin_val": 0.2,
                    "lot_value": 20000.0, "min_lot": 1, "lot_step": 1, "lot_max": 10000,
                    "leverage": 20, "margin_rate": 0.05,
                    "commissions": 0.5, "slippage": 0.25, "spread": 0.25,
                    "swap_long": 0.0, "swap_short": 0.0,
                    "datetime_candle_references": "open",
                },
                "WDO$": {
                    "tick": 5, "tick_fin_val": 5.0,
                    "lot_value": 40000.0, "min_lot": 1, "lot_step": 1, "lot_max": 10000,
                    "leverage": 20, "margin_rate": 0.05,
                    "commissions": 0.5, "slippage": 0.25, "spread": 0.25,
                    "swap_long": 0.0, "swap_short": 0.0,
                    "datetime_candle_references": "open",
                },
            },
        },
        "currency_pair": {
            "forex": {
                "generic": {
                    "tick": 0.0001, "tick_fin_val": 10,
                    "lot_value": 100000.0, "min_lot": 0.01, "lot_step": 0.01, "lot_max": 500,
                    "leverage": 100, "margin_rate": 0.01,
                    "commissions": 1.5, "slippage": 0.75, "spread": 0.75,
                    "swap_long": -0.5, "swap_short": -0.5,
                    "datetime_candle_references": "open",
                },
            },
        },
        "stock": {
            "NASDAQ": {
                "generic": {
                    "tick": 0.01, "tick_fin_val": 1,
                    "lot_value": 100, "min_lot": 1, "lot_step": 1, "lot_max": 10000,
                    "leverage": 1, "margin_rate": 1.0,
                    "commissions": 5.0, "slippage": 0.05, "spread": 0.02,
                    "swap_long": 0.0, "swap_short": 0.0,
                    "datetime_candle_references": "open",
                },
            },
        },
    }

    _DEFAULT_PARAMS: Dict = {
        "tick": 0.01, "tick_fin_val": 1.0,
        "lot_value": 100.0, "min_lot": 1.0, "lot_step": 1.0, "lot_max": 10000.0,
        "leverage": 1.0, "margin_rate": 1.0,
        "commissions": 0.0, "slippage": 0.0, "spread": 0.0,
        "swap_long": 0.0, "swap_short": 0.0,
        "datetime_candle_references": "open",
    }

    def __init__(
        self,
        name:       str,
        type:       str,
        market:     str,
        date_start: Optional[str] = None,
        date_end:   Optional[str] = None,
        **kwargs,
    ):
        self.name       = name
        self.type       = type
        self.market     = market
        self.date_start = date_start
        self.date_end   = date_end

        self._cache: Dict[str, pl.DataFrame] = {}   # {timeframe: df}

        # Parâmetros de mercado
        params = self._load_params()
        params.update(kwargs)
        for k, v in params.items():
            setattr(self, k, v)

    # ── Market params ─────────────────────────────────────────────────────────

    def _load_params(self) -> Dict:
        try:
            return self.ASSET_PARAMS[self.type][self.market][self.name].copy()
        except KeyError:
            pass
        try:
            return self.ASSET_PARAMS[self.type][self.market]["generic"].copy()
        except KeyError:
            pass
        return self._DEFAULT_PARAMS.copy()

    # ── Load (Strategy Pattern) ───────────────────────────────────────────────

    def load(
        self,
        timeframe:  str,
        source:     SourceType = "local",
        date_start: Optional[str] = None,
        date_end:   Optional[str] = None,
        cache:      bool = True,
        **kwargs,
    ) -> pl.DataFrame:
        """
        Carrega dados de qualquer source e armazena em cache.

        Parâmetros
        ----------
        timeframe  : ex. 'M15', 'D1'
        source     : 'local' | 'file' | 'yfinance' | 'fred'
        date_start : filtra a partir desta data ('YYYY-MM-DD')
        date_end   : filtra até esta data ('YYYY-MM-DD')
        cache      : armazena em memória após carregar
        **kwargs   : repassados para a DataSource (ex: path=, delimiter=)

        Exemplos
        --------
        asset.load('M15')
        asset.load('D1', source='yfinance', date_start='2020-01-01')
        asset.load('M15', source='file', path='raw/EURUSD_M15.csv')
        """
        ds  = date_start or self.date_start
        de  = date_end   or self.date_end
        key = f"{timeframe}_{source}_{ds}_{de}"

        if cache and key in self._cache:
            return self._cache[key]

        handler = _SOURCE_REGISTRY.get(source)
        if handler is None:
            raise ValueError(f"Unknown source '{source}'. Available: {list(_SOURCE_REGISTRY)}")

        df = handler.load(
            name=self.name,
            timeframe=timeframe,
            date_start=ds,
            date_end=de,
            market=self.market,
            **kwargs,
        )

        if cache:
            self._cache[key] = df

        return df

    @staticmethod
    def load_all(base_path: str = None) -> Dict[str, "Asset"]:
        """
        Lê todos os registries em Backend/Assets/{market}/registry.json
        e retorna um dict {name: Asset} pronto para uso.

        Uso
        ---
            assets = Asset.load_all()
            assets["EURUSD"].data_get("M15")
            assets["GBPUSD"].load("D1", date_start="2022-01-01")
        """
        root = Path(base_path) if base_path else ASSETS_BASE
        result: Dict[str, Asset] = {}

        for registry_file in sorted(root.rglob("registry.json")):
            market = registry_file.parent.name

            with open(registry_file, "r") as f:
                data = json.load(f)

            asset_type = data.get("asset_type", "")

            for name, meta in data.get("assets", {}).items():
                params = meta.get("asset_params", {})
                result[name] = Asset(
                    name=name,
                    type=asset_type,
                    market=market,
                    **params,
                )

        print(f"   > [Asset] {len(result)} assets loaded from registry")
        return result

    # ── data_get: compatibilidade com Operation ───────────────────────────────

    def data_get(self, timeframe: str) -> pl.DataFrame:
        """Mantém interface esperada pelo Operation/Walkforward."""
        return self.load(timeframe, source="local")

    def set_date_range(self, date_start: Optional[str] = None, date_end: Optional[str] = None):
        self.date_start = date_start
        self.date_end   = date_end
        self._cache.clear()

    # ── Convert: importa arquivo e salva parquet + registry ──────────────────

    def convert(
        self,
        source_path:   str,
        timeframe:     str       = None,
        datetime_col:  str       = None,
        datetime_fmt:  str       = None,
        delimiter:     str       = ",",
        update_reason: str       = "file import",
    ) -> Optional[Path]:
        """
        Importa CSV/XLSX, salva parquet em Backend/Assets/{market}/
        e atualiza o registry.json do market.

        O timeframe é inferido do nome do arquivo se não passado.
        """
        source = Path(source_path)
        if not source.exists():
            print(f"   ⚠️ File not found: {source}")
            return None

        # Infere timeframe do nome do arquivo se não passado
        if not timeframe:
            m = re.search(r"_([A-Z][0-9]+)", source.stem, re.IGNORECASE)
            timeframe = m.group(1).upper() if m else "UNKNOWN"

        # Usa FileSource para carregar + normalizar
        df = FileSource().load(
            name=self.name,
            timeframe=timeframe,
            market=self.market,
            path=str(source),
            datetime_col=datetime_col,
            datetime_format=datetime_fmt,
            delimiter=delimiter,
            save=True,
        )

        # Atualiza registry
        registry = AssetRegistry(market=self.market, asset_type=self.type)
        registry.upsert(
            name=self.name,
            timeframe=timeframe,
            df=df,
            asset_params=self._load_params(),
            update_reason=update_reason,
        )

        # Valida
        out_file = ASSETS_BASE / self.market / f"{self.name}_{timeframe}.parquet"
        self._validate(out_file, df, timeframe)

        return out_file

    def _validate(self, path: Path, original: pl.DataFrame, timeframe: str):
        loaded  = pl.read_parquet(path)
        ok_rows = len(loaded) == len(original)
        nulls   = {c: loaded[c].null_count() for c in loaded.columns if loaded[c].null_count() > 0}
        status  = "✅ OK" if ok_rows and not nulls else "⚠️  ISSUES"

        print(f"\n{'─'*60}")
        print(f"  {status}  {self.name}_{timeframe}.parquet")
        print(f"{'─'*60}")
        print(f"  Rows   : {len(loaded):,}  {'✅' if ok_rows else '❌ MISMATCH (original: ' + str(len(original)) + ')'}")
        print(f"  Range  : {loaded['datetime'].min()}  →  {loaded['datetime'].max()}")
        print(f"  Cols   : {loaded.columns}")
        print(f"  Nulls  : {nulls if nulls else 'none ✅'}")
        print(f"  Size   : {path.stat().st_size / 1024:.1f} KB")
        print(f"{'─'*60}\n")

    # ── Utils ─────────────────────────────────────────────────────────────────

    @property
    def timeframes(self) -> List[str]:
        """Timeframes disponíveis em cache."""
        return list(self._cache.keys())

    def __repr__(self) -> str:
        return (f"<Asset {self.name} | {self.type}/{self.market} | "
                f"tick={self.tick} | date: {self.date_start or 'all'} → {self.date_end or 'all'}>")

# ═════════════════════════════════════════════════════════════════════════════
# CONVERT FOLDER  (utility standalone)
# ═════════════════════════════════════════════════════════════════════════════

def convert_folder(
    source_folder: str,
    asset_type:    str,
    market:        str,
    datetime_col:  str = None,
    datetime_fmt:  str = None,
    delimiter:     str = ",",
    update_reason: str = "bulk folder import",
):    
    # Converte todos os CSV/XLSX de uma pasta para parquet.
    # Uso
    #     convert_folder("raw/Forex", asset_type="currency_pair", market="forex")
    
    folder = Path(source_folder)
    if not folder.is_dir():
        print(f"⚠️ Folder not found: {folder}")
        return

    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in (".csv", ".xlsx", ".xls"))
    if not files:
        print(f"⚠️ No CSV/XLSX files in {folder}")
        return

    print(f"\n>>> Converting {len(files)} file(s) from '{folder}'\n")

    for file in files:
        m          = re.match(r"([A-Za-z0-9$]+?)(?:_[A-Z][0-9]+)?$", file.stem, re.IGNORECASE)
        asset_name = m.group(1).upper() if m else file.stem.upper()
        print(f"  → {file.name}  (asset: {asset_name})")

        asset = Asset(name=asset_name, type=asset_type, market=market)
        asset.convert(
            source_path=str(file),
            datetime_col=datetime_col,
            datetime_fmt=datetime_fmt,
            delimiter=delimiter,
            update_reason=update_reason,
        )

    # Mostra registry final do market
    AssetRegistry(market=market, asset_type=asset_type).summary()









