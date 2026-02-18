import polars as pl
import os
import re
import datetime
from BaseClass import BaseClass
from dataclasses import dataclass
from typing import Union, Callable, Dict, List, Optional

# =================================================================================================================================|| UTILS

def create_datetime_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Garante que o DataFrame tenha colunas 'datetime', 'date' e 'time' usando Polars."""
    
    # 1. Tenta criar a coluna 'datetime' se ela não existir
    if 'datetime' not in df.columns:
        if 'date' in df.columns and 'time' in df.columns:
            # Caso data e hora separados (Trata strings ou tipos data)
            df = df.with_columns([
                pl.format("{} {}", pl.col("date"), pl.col("time"))
                .str.to_datetime(strict=False)
                .alias("datetime")
            ])
        elif 'date' in df.columns:
            df = df.with_columns([
                pl.col("date").cast(pl.Utf8).str.to_datetime(strict=False).alias("datetime")
            ])
        elif 'time' in df.columns:
            # Apenas hora: assume uma data base (1900-01-01)
            df = df.with_columns([
                pl.format("1900-01-01 {}", pl.col("time"))
                .str.to_datetime(strict=False)
                .alias("datetime")
            ])
        else:
            raise ValueError("Não foi possível criar 'datetime': faltam colunas 'date' e 'time'.")

    # 2. Garante consistência nas colunas complementares
    # No Polars, extraímos propriedades de data/hora via .dt
    df = df.with_columns([
        pl.col("datetime").dt.date().alias("date"),
        pl.col("datetime").dt.time().alias("time")
    ])

    # 3. Remove nulos e reseta o "índice" (no Polars apenas drop_nulls)
    df = df.drop_nulls(subset=["datetime"])
    
    return df

def load_data(data_path, min_limit=0.0, max_limit=1.0, index_col_setting=False, drop_index=True):
    """Carrega dados usando Polars com suporte a CSV e Excel."""
    
    def normalize_columns(df: pl.DataFrame):
        # Converte todas as colunas para lowercase
        return df.select([pl.col(c).alias(c.lower()) for c in df.columns])

    def process_file(file_path):
        try:
            # 1. Leitura inicial
            if file_path.endswith(('.xlsx', '.xls')):
                df = pl.read_excel(file_path)
            elif file_path.endswith('.csv'):
                df = pl.read_csv(file_path)
            else:
                return None

            df = normalize_columns(df)

            # 2. NOVO COMPONENTE: TRATAMENTO ROBUSTO DE DATETIME
            if "datetime" not in df.columns:
                if "date" in df.columns and "time" in df.columns:
                    # Convertemos tudo para String primeiro para evitar o erro de tipos diferentes
                    df = df.with_columns([
                        pl.col("date").cast(pl.Utf8),
                        pl.col("time").cast(pl.Utf8)
                    ])
                    
                    # Agora concatenamos com segurança
                    df = df.with_columns(
                        (pl.col("date") + pl.lit(" ") + pl.col("time")).alias("datetime")
                    )
                else:
                    raise ValueError(f"Colunas temporais não encontradas em {file_path}")

            # 3. CONVERSÃO PARA DATETIME (Tratando formatos de Excel vs CSV)
            # O Excel costuma usar 'YYYY-MM-DD', o CSV do MT5 costuma usar 'YYYY.MM.DD'
            df = df.with_columns(
                pl.col("datetime").str.replace_all(r"\.", "-").str.to_datetime(strict=False)
            )

            # 4. GARANTIA DE SUCESSO
            if df["datetime"].null_count() == len(df):
                raise ValueError("Falha crítica: a conversão de datetime resultou apenas em valores nulos.")

            # Restante do código (ativo name, slice, etc)
            ativo_name = os.path.splitext(os.path.basename(file_path))[0]
            df = df.with_columns(pl.lit(ativo_name).alias('ativo'))
            
            # Chamada da sua função de colunas extras (se existir)
            # df = create_datetime_columns(df)

            total_rows = len(df)
            start_idx = int(total_rows * min_limit)
            end_idx = int(total_rows * max_limit)
            
            return df.slice(start_idx, end_idx - start_idx)

        except Exception as e:
            print(f"Erro ao processar {file_path}: {str(e)}")
            # Retornamos None para que o load_data filtre e não envie um DF vazio
            return None

    # Lógica de diretório
    if os.path.isdir(data_path):
        files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
                if f.endswith(('.xlsx', '.xls', '.csv'))]
        results = [process_file(f) for f in files]
        return [df for df in results if df is not None]
    
    # Lógica de arquivo único com tentativa de extensões
    possible_extensions = ['.xlsx', '.xls', '.csv']
    
    if any(data_path.endswith(ext) for ext in possible_extensions):
        result = process_file(data_path)
        return result if result is not None else pl.DataFrame()
    
    for ext in possible_extensions:
        file_path = data_path + ext
        if os.path.exists(file_path):
            result = process_file(file_path)
            if result is not None:
                return result
    
    return pl.DataFrame()

# =================================================================================================================================|| ASSET

@dataclass
class AssetParams:
    tick: float = 0.01
    tick_fin_val: float = 1.0
    lot_value: float = 100.0
    min_lot: float = 1.0
    leverage: float = 1.0
    commissions: float = 0.0
    slippage: float = 0.0
    spread: float = 0.0
    datetime_candle_references: str = 'open'

class Asset(BaseClass):
    ASSET_PARAMS = {
        'futures': {
            'b3': {
                'WIN$': {
                    'tick': 1, 'tick_fin_val': 0.2, 'lot_value': 20000.0, 'min_lot': 1,
                    'leverage': 20, 'commissions': 0.5, 'slippage': 0.25, 'spread': 0.25,
                    'datetime_candle_references': 'open'
                },
                'WDO$': {
                    'tick': 5, 'tick_fin_val': 5.0, 'lot_value': 40000.0, 'min_lot': 1,
                    'leverage': 20, 'commissions': 0.5, 'slippage': 0.25, 'spread': 0.25,
                    'datetime_candle_references': 'open'
                }
            }
        },
        'currency_pair': {
            'forex': {
                'generic': {
                    'tick': 0.0001, 'tick_fin_val': 10, 'lot_value': 100000.0, 'min_lot': 0.01,
                    'leverage': 100, 'commissions': 1.5, 'slippage': 0.75, 'spread': 0.75,
                    'datetime_candle_references': 'open'
                }
            }
        },
        'stock': {
            'NASDAQ': {'generic': {'tick': 0.01, 'tick_fin_val': 1, 'lot_value': 100, 'min_lot': 1, 'leverage': 1, 'commissions': 5.0, 'slippage': 0.05, 'spread': 0.02, 'datetime_candle_references': 'open'}},
            'NYSE': {'generic': {'tick': 0.01, 'tick_fin_val': 1, 'lot_value': 100, 'min_lot': 1, 'leverage': 1, 'commissions': 5.0, 'slippage': 0.05, 'spread': 0.02, 'datetime_candle_references': 'open'}}
        }
    }

    def __init__(self, name: str, type: str, market: str, timeframe: List[str] = None, 
                 data_path: str = 'C:\\Users\\Patrick\\Desktop\\Artaxes Portfolio\\MAIN\\MT5_Dados', **kwargs):
        super().__init__()
        self.name = name
        self.type = type
        self.market = market
        self.data_path = data_path
        self.data: Dict[str, Optional[pl.DataFrame]] = {}
        self.timeframe = timeframe
        
        if timeframe:
            for tf in timeframe: self.data[tf] = None
        else:
            self.timeframes_load_available()

        default_params = self.get_default_params()
        for key, value in {**default_params, **kwargs}.items():
            setattr(self, key, value)

    def get_default_params(self):
        try:
            return self.ASSET_PARAMS[self.type][self.market][self.name].copy()
        except KeyError:
            try:
                return self.ASSET_PARAMS[self.type][self.market]['generic'].copy()
            except KeyError:
                return { 
                    'tick': 0.01, 'tick_fin_val': 1.0, 'lot_value': 100.0, 'min_lot': 1.0,
                    'leverage': 1.0, 'commissions': 0.0, 'slippage': 0.0, 'spread': 0.0,
                    'datetime_candle_references': 'open'
                }

    def data_get(self, timeframe: str) -> pl.DataFrame:
        if timeframe not in self.data or self.data[timeframe] is None:
            data_file = os.path.join(self.data_path, f"{self.name}_{timeframe}")
            self.data[timeframe] = load_data(data_file)
        return self.data[timeframe]

    def timeframes_load_available(self):
        if not os.path.isdir(self.data_path): return
        pattern = re.compile(f"{self.name}_([A-Z][0-9]+)")
        for file in os.listdir(self.data_path):
            match = pattern.match(file)
            if match:
                tf = match.group(1)
                if tf not in self.data: self.data[tf] = None

    @staticmethod
    def load_unique_assets(assets: Dict[str, "Asset"], type: str, market: str, path: str) -> Dict[str, "Asset"]:
        if not os.path.isdir(path): return assets
        pattern = re.compile(r"([A-Za-z0-9$]+)_[A-Z][0-9]+\.(csv|xlsx|xls)")
        for file in os.listdir(path):
            match = pattern.match(file)
            if match:
                asset_name = match.group(1)
                if asset_name not in assets:
                    assets[asset_name] = Asset(asset_name, type, market, data_path=path)
        return assets

# =================================================================================================================================|| PORTFOLIO

class Asset_Portfolio(BaseClass):
    def __init__(self, asset_portfolio_params: dict = None, name: str = None, assets: Union[dict, list, None] = None):
        super().__init__()
        if asset_portfolio_params:
            self.name = asset_portfolio_params.get('name', 'unnamed_portfolio')
            assets_input = asset_portfolio_params.get('assets', {})
        else:
            self.name = name or 'unnamed_portfolio'
            assets_input = assets or {}

        self.assets: Dict[str, Asset] = {}
        if isinstance(assets_input, dict):
            self.assets = assets_input
        elif isinstance(assets_input, list):
            for a in assets_input:
                if isinstance(a, Asset): self.assets[a.name] = a

    def data_get(self, timeframes: List[str]) -> Dict[str, Dict[str, pl.DataFrame]]:
        result = {}
        for tf in timeframes:
            result[tf] = {}
            for name, asset in self.assets.items():
                try:
                    result[tf][name] = asset.data_get(tf)
                except Exception as e:
                    print(f"Error loading {name} @ {tf}: {e}")
                    result[tf][name] = pl.DataFrame()
        return result

    def calculate_correlation(self, timeframe: str) -> pl.DataFrame:
        """Calcula correlação entre ativos usando o preço de fechamento."""
        combined = None
        for name, asset in self.assets.items():
            df = asset.data_get(timeframe).select([
                pl.col("datetime"),
                pl.col("close").alias(name)
            ])
            if combined is None:
                combined = df
            else:
                combined = combined.join(df, on="datetime", how="outer")
        
        if combined is None: return pl.DataFrame()
        
        # Polars correlation matrix
        numeric_cols = [c for c in combined.columns if c != "datetime"]
        return combined.select(numeric_cols).corr()

    def preload_data(self, timeframes: List[str], verbose: bool = False) -> None:
        if verbose: print(f"\nPrecarregando dados para {len(self.assets)} assets...")
        for name, asset in self.assets.items():
            for tf in timeframes:
                df = asset.data_get(tf)
                if verbose: print(f"Asset {name} - TF {tf}: OK ({len(df)} linhas)")