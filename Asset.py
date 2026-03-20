import polars as pl, os, re
from pathlib import Path
from typing import Union, Dict, List, Optional


# ═════════════════════════════════════════════════════════════════════════════
# UTILS
# ═════════════════════════════════════════════════════════════════════════════

def load_data(data_path: str) -> pl.DataFrame:
    """
    Carrega dados de CSV ou Excel para Polars DataFrame.
    Suporta arquivo único (com ou sem extensão) ou diretório.
    Normaliza colunas para lowercase e converte datetime automaticamente.
    """
    def _process_file(file_path: str) -> Optional[pl.DataFrame]:
        try:
            if file_path.endswith(('.xlsx', '.xls')):
                df = pl.read_excel(file_path)
            elif file_path.endswith('.csv'):
                df = pl.read_csv(file_path)
            else:
                return None

            # Normaliza colunas para lowercase
            df = df.rename({c: c.lower() for c in df.columns})

            # Monta coluna datetime se necessário
            if "datetime" not in df.columns:
                if "date" in df.columns and "time" in df.columns:
                    df = df.with_columns([
                        pl.col("date").cast(pl.Utf8),
                        pl.col("time").cast(pl.Utf8),
                    ]).with_columns(
                        (pl.col("date") + pl.lit(" ") + pl.col("time")).alias("datetime")
                    )
                else:
                    raise ValueError(f"No datetime/date+time columns found in {file_path}")

            # Converte datetime — suporta MT5 (YYYY.MM.DD) e ISO (YYYY-MM-DD)
            df = df.with_columns(
                pl.col("datetime")
                .str.replace_all(r"\.", "-")
                .str.to_datetime(strict=False)
            )

            if df["datetime"].null_count() == len(df):
                raise ValueError(f"Datetime conversion failed for {file_path}")

            # Nome do ativo a partir do filename
            asset_name = os.path.splitext(os.path.basename(file_path))[0]
            df = df.with_columns(pl.lit(asset_name).alias("ativo"))

            return df

        except Exception as e:
            print(f"   ⚠️ Error loading {file_path}: {e}")
            return None

    path = Path(data_path)

    # Diretório — carrega todos os arquivos
    if path.is_dir():
        dfs = []
        for f in sorted(path.iterdir()):
            if f.suffix in ('.csv', '.xlsx', '.xls'):
                df = _process_file(str(f))
                if df is not None:
                    dfs.append(df)
        return pl.concat(dfs, how="diagonal_relaxed") if dfs else pl.DataFrame()

    # Arquivo único — tenta com e sem extensão
    if path.exists():
        result = _process_file(str(path))
        return result if result is not None else pl.DataFrame()

    for ext in ('.csv', '.xlsx', '.xls'):
        candidate = Path(str(data_path) + ext)
        if candidate.exists():
            result = _process_file(str(candidate))
            return result if result is not None else pl.DataFrame()

    return pl.DataFrame()


# ═════════════════════════════════════════════════════════════════════════════
# ASSET
# ═════════════════════════════════════════════════════════════════════════════

class Asset:
    """
    Representa um ativo financeiro com seus parâmetros de mercado e dados OHLC.

    Parâmetros de mercado (tick, tick_fin_val, lot_*, leverage, custos) são
    carregados automaticamente do ASSET_PARAMS via type/market/name.
    Parâmetros extras podem ser passados via **kwargs para sobrescrever.

    Filtro de data: date_start e date_end filtram os dados retornados por data_get().
    """

    ASSET_PARAMS: Dict = {
        'futures': {
            'b3': {
                'WIN$': {
                    'tick': 1, 'tick_fin_val': 0.2,
                    'lot_value': 20000.0, 'min_lot': 1, 'lot_step': 1, 'lot_max': 10000,
                    'leverage': 20, 'margin_rate': 0.05,
                    'commissions': 0.5, 'slippage': 0.25, 'spread': 0.25,
                    'swap_long': 0.0, 'swap_short': 0.0,
                    'datetime_candle_references': 'open',
                },
                'WDO$': {
                    'tick': 5, 'tick_fin_val': 5.0,
                    'lot_value': 40000.0, 'min_lot': 1, 'lot_step': 1, 'lot_max': 10000,
                    'leverage': 20, 'margin_rate': 0.05,
                    'commissions': 0.5, 'slippage': 0.25, 'spread': 0.25,
                    'swap_long': 0.0, 'swap_short': 0.0,
                    'datetime_candle_references': 'open',
                },
            },
        },
        'currency_pair': {
            'forex': {
                'generic': {
                    'tick': 0.0001, 'tick_fin_val': 10,
                    'lot_value': 100000.0, 'min_lot': 0.01, 'lot_step': 0.01, 'lot_max': 500,
                    'leverage': 100, 'margin_rate': 0.01,
                    'commissions': 1.5, 'slippage': 0.75, 'spread': 0.75,
                    'swap_long': -0.5, 'swap_short': -0.5,
                    'datetime_candle_references': 'open',
                },
            },
        },
        'stock': {
            'NASDAQ': {
                'generic': {
                    'tick': 0.01, 'tick_fin_val': 1,
                    'lot_value': 100, 'min_lot': 1, 'lot_step': 1, 'lot_max': 10000,
                    'leverage': 1, 'margin_rate': 1.0,
                    'commissions': 5.0, 'slippage': 0.05, 'spread': 0.02,
                    'swap_long': 0.0, 'swap_short': 0.0,
                    'datetime_candle_references': 'open',
                },
            },
            'NYSE': {
                'generic': {
                    'tick': 0.01, 'tick_fin_val': 1,
                    'lot_value': 100, 'min_lot': 1, 'lot_step': 1, 'lot_max': 10000,
                    'leverage': 1, 'margin_rate': 1.0,
                    'commissions': 5.0, 'slippage': 0.05, 'spread': 0.02,
                    'swap_long': 0.0, 'swap_short': 0.0,
                    'datetime_candle_references': 'open',
                },
            },
        },
    }

    # Defaults quando asset não encontrado no ASSET_PARAMS
    _DEFAULT_PARAMS: Dict = {
        'tick': 0.01, 'tick_fin_val': 1.0,
        'lot_value': 100.0, 'min_lot': 1.0, 'lot_step': 1.0, 'lot_max': 10000.0,
        'leverage': 1.0, 'margin_rate': 1.0,
        'commissions': 0.0, 'slippage': 0.0, 'spread': 0.0,
        'swap_long': 0.0, 'swap_short': 0.0,
        'datetime_candle_references': 'open',
    }

    def __init__(
        self,
        name:        str,
        type:        str,
        market:      str,
        data_path:   str = '',
        timeframe:   Optional[List[str]] = None,
        date_start:  Optional[str] = None,   # 'YYYY-MM-DD'
        date_end:    Optional[str] = None,   # 'YYYY-MM-DD'
        **kwargs,
    ):
        self.name       = name
        self.type       = type
        self.market     = market
        self.data_path  = data_path
        self.date_start = date_start
        self.date_end   = date_end

        self._data: Dict[str, Optional[pl.DataFrame]] = {}

        # Inicializa timeframes disponíveis
        if timeframe:
            for tf in timeframe:
                self._data[tf] = None
        else:
            self._discover_timeframes()

        # Aplica parâmetros de mercado — ASSET_PARAMS → generic → defaults → kwargs
        params = self._load_params()
        params.update(kwargs)  # kwargs sobrescreve tudo
        for k, v in params.items():
            setattr(self, k, v)

    # ─────────────────────────────────────────────────────────────────────────
    # Params
    # ─────────────────────────────────────────────────────────────────────────

    def _load_params(self) -> Dict:
        """Carrega parâmetros de mercado com fallback: específico → generic → defaults."""
        try:
            return self.ASSET_PARAMS[self.type][self.market][self.name].copy()
        except KeyError:
            pass
        try:
            return self.ASSET_PARAMS[self.type][self.market]['generic'].copy()
        except KeyError:
            pass
        return self._DEFAULT_PARAMS.copy()

    # ─────────────────────────────────────────────────────────────────────────
    # Data
    # ─────────────────────────────────────────────────────────────────────────

    def data_get(self, timeframe: str) -> pl.DataFrame:
        """
        Retorna dados OHLC para o timeframe especificado.
        Carrega do disco se ainda não em cache.
        Aplica filtro de date_start/date_end se definidos no Asset.
        """
        if timeframe not in self._data or self._data[timeframe] is None:
            file_path = os.path.join(self.data_path, f"{self.name}_{timeframe}")
            df = load_data(file_path)

            if df.is_empty():
                print(f"   ⚠️ WARNING: No data found for {self.name} @ {timeframe}")
                self._data[timeframe] = df
            else:
                self._data[timeframe] = df

        df = self._data[timeframe]

        if df is None or df.is_empty():
            return pl.DataFrame()

        # Aplica filtro de data
        df = self._filter_dates(df)

        # Warnings de cobertura
        self._check_date_coverage(df, timeframe)

        return df

    def _filter_dates(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filtra o DataFrame pelo date_start e date_end do Asset."""
        if not self.date_start and not self.date_end:
            return df

        dt = pl.col("datetime")
        if self.date_start:
            start = pl.lit(self.date_start).str.to_datetime("%Y-%m-%d")
            df = df.filter(dt >= start)
        if self.date_end:
            end = pl.lit(self.date_end).str.to_datetime("%Y-%m-%d")
            df = df.filter(dt <= end)

        return df

    def _check_date_coverage(self, df: pl.DataFrame, timeframe: str):
        """Emite warnings se o dado não cobre o range solicitado."""
        if df.is_empty():
            print(f"   ⚠️ WARNING: {self.name} @ {timeframe} has no data in range "
                  f"{self.date_start or 'start'} → {self.date_end or 'end'}")
            return

        actual_start = df["datetime"].min()
        actual_end   = df["datetime"].max()

        if self.date_start:
            expected = pl.Series([self.date_start]).str.to_datetime("%Y-%m-%d")[0]
            gap = abs((actual_start - expected).days) if hasattr((actual_start - expected), 'days') else 0
            if gap > 5:
                print(f"   ⚠️ WARNING: {self.name} @ {timeframe} starts at {actual_start} "
                      f"(expected {self.date_start}, {gap}d gap)")

        if self.date_end:
            expected = pl.Series([self.date_end]).str.to_datetime("%Y-%m-%d")[0]
            gap = (expected - actual_end).days if hasattr((expected - actual_end), 'days') else 0
            if gap > 5:
                print(f"   ⚠️ WARNING: {self.name} @ {timeframe} ends at {actual_end} "
                      f"(expected {self.date_end}, {gap}d missing)")

    def set_date_range(self, date_start: Optional[str] = None, date_end: Optional[str] = None):
        """
        Atualiza o filtro de data e invalida o cache para recarregar.
        Útil para reutilizar o mesmo Asset em múltiplas Operations com ranges diferentes.
        """
        self.date_start = date_start
        self.date_end   = date_end

    def _discover_timeframes(self):
        """Descobre timeframes disponíveis no data_path pelo padrão {NAME}_{TF}."""
        if not self.data_path or not os.path.isdir(self.data_path):
            return
        pattern = re.compile(rf"{re.escape(self.name)}_([A-Z][0-9]+)", re.IGNORECASE)
        for file in os.listdir(self.data_path):
            match = pattern.match(file)
            if match:
                tf = match.group(1).upper()
                if tf not in self._data:
                    self._data[tf] = None

    @property
    def timeframes(self) -> List[str]:
        """Lista de timeframes disponíveis."""
        return list(self._data.keys())

    # ─────────────────────────────────────────────────────────────────────────
    # Utils
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def load_unique_assets(
        assets:  Dict[str, "Asset"],
        type:    str,
        market:  str,
        path:    str,
    ) -> Dict[str, "Asset"]:
        """Carrega automaticamente todos os ativos únicos de um diretório."""
        if not os.path.isdir(path):
            return assets
        pattern = re.compile(r"([A-Za-z0-9$]+)_[A-Z][0-9]+\.(csv|xlsx|xls)", re.IGNORECASE)
        for file in os.listdir(path):
            match = pattern.match(file)
            if match:
                asset_name = match.group(1)
                if asset_name not in assets:
                    assets[asset_name] = Asset(asset_name, type, market, data_path=path)
        return assets

    def __repr__(self) -> str:
        tfs = ', '.join(self.timeframes) or 'none'
        return (f"<Asset {self.name} | {self.type}/{self.market} | "
                f"tick={self.tick} tick_fin_val={self.tick_fin_val} | "
                f"TFs: {tfs} | "
                f"date: {self.date_start or 'all'} → {self.date_end or 'all'}>")


# ═════════════════════════════════════════════════════════════════════════════
# ASSET PORTFOLIO
# ═════════════════════════════════════════════════════════════════════════════

class AssetPortfolio:
    """
    Coleção de Assets operados em conjunto.
    Fornece acesso unificado a dados e correlações.
    """

    def __init__(
        self,
        name:   str = 'portfolio',
        assets: Union[Dict[str, Asset], List[Asset], None] = None,
    ):
        self.name   = name
        self.assets: Dict[str, Asset] = {}

        if isinstance(assets, dict):
            self.assets = assets
        elif isinstance(assets, list):
            for a in assets:
                if isinstance(a, Asset):
                    self.assets[a.name] = a

    def data_get(self, timeframes: List[str]) -> Dict[str, Dict[str, pl.DataFrame]]:
        """Retorna {tf: {asset_name: df}} para todos os assets e timeframes."""
        result: Dict[str, Dict[str, pl.DataFrame]] = {}
        for tf in timeframes:
            result[tf] = {}
            for name, asset in self.assets.items():
                try:
                    result[tf][name] = asset.data_get(tf)
                except Exception as e:
                    print(f"   ⚠️ Error loading {name} @ {tf}: {e}")
                    result[tf][name] = pl.DataFrame()
        return result

    def set_date_range(self, date_start: Optional[str] = None, date_end: Optional[str] = None):
        """Propaga filtro de data para todos os assets do portfolio."""
        for asset in self.assets.values():
            asset.set_date_range(date_start, date_end)

    def calculate_correlation(self, timeframe: str) -> pl.DataFrame:
        """Matriz de correlação entre assets pelo preço de fechamento."""
        combined = None
        for name, asset in self.assets.items():
            df = asset.data_get(timeframe).select([
                pl.col("datetime"),
                pl.col("close").alias(name),
            ])
            combined = df if combined is None else combined.join(df, on="datetime", how="full", coalesce=True)

        if combined is None:
            return pl.DataFrame()

        numeric_cols = [c for c in combined.columns if c != "datetime"]
        return combined.select(numeric_cols).corr()

    def preload_data(self, timeframes: List[str], verbose: bool = False):
        """Pré-carrega dados de todos os assets para os timeframes especificados."""
        if verbose:
            print(f"\nPreloading data for {len(self.assets)} assets...")
        for name, asset in self.assets.items():
            for tf in timeframes:
                df = asset.data_get(tf)
                if verbose:
                    print(f"   {name} @ {tf}: {len(df)} rows")

    def __repr__(self) -> str:
        return f"<AssetPortfolio '{self.name}' | {len(self.assets)} assets: {list(self.assets.keys())}>"