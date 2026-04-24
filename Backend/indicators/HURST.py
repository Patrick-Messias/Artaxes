import polars as pl, numpy as np
from typing import Union
from Backend.core.Indicator import Indicator

class HURST(Indicator):
    # Indicador Hurst Exponent.
    # Avalia se a série tem comportamento de reversão à média (H < 0.5), 
    # passeio aleatório (H = 0.5) ou tendência (H > 0.5).
    #
    # Métodos:
    #     "simple" -> Rescaled Range simplificado.
    #     "rs"     -> Rescaled Range clássico com polyfit.

    def __init__(self, asset=None, timeframe=None, **params):
        defaults = {
            'window': 63, 
            'calc_type': 'simple',      # Mudado de 'type' para evitar conflito com palavra reservada do Python
            'data_target': 'pct_change', # O que calcular: 'close', 'pct_change', 'log_returns'
            'price_col': 'close'         # Coluna base para puxar do DataFrame
        }
        defaults.update(params)
        super().__init__(asset, timeframe, **defaults)
        self.name = "hurst"

    def _calculate_logic(self, data: Union[pl.DataFrame, pl.Series], **kwargs) -> pl.Series:
        window = int(kwargs.get('window', 63))
        calc_type = str(kwargs.get('calc_type', 'simple')).lower()
        data_target = str(kwargs.get('data_target', 'pct_change')).lower()
        price_col = str(kwargs.get('price_col', 'close'))

        # 1. Extração da Série base (idêntico ao VAR)
        if isinstance(data, pl.Series):
            s = data
        elif isinstance(data, pl.DataFrame):
            if price_col not in data.columns:
                price_col = next((c for c in data.columns if c.lower() not in ['ts', 'datetime']), data.columns[0])
            s = data.get_column(price_col)
        else:
            raise ValueError(f"HURST: Type not supported: {type(data)}")

        # 2. Transformação dos Dados (Retornos, Log, ou Preço Puro)
        if data_target == 'pct_change':
            s = s.pct_change().fill_null(0.0)
        elif data_target == 'log_returns':
            # log_returns = ln(P_t / P_{t-1})
            s = s.log().diff().fill_null(0.0)
        elif data_target == 'close':
            pass # Usa a série pura
        else:
            raise ValueError(f"HURST: Transformação não suportada: {data_target}")

        # 3. Funções Numéricas Otimizadas para a Janela
        def simple_hurst(window_series: pl.Series) -> float:
            arr = window_series.to_numpy()
            n = len(arr)
            if n < 2: return float('nan')
            
            deviations = arr - np.mean(arr)
            cumulative_dev = np.cumsum(deviations)
            R = np.max(cumulative_dev) - np.min(cumulative_dev)
            S = np.std(arr, ddof=1)
            
            if S == 0 or R == 0: return 0.5
            return float(np.log(R/S) / np.log(n))

        def hurst_rs(window_series: pl.Series) -> float:
            arr = window_series.to_numpy()
            n = len(arr)
            min_lag, max_lag = 2, 20
            lags = range(min_lag, min(max_lag, n // 2))
            
            if len(lags) < 2: return float('nan')
            
            tau = [np.std(np.subtract(arr[lag:], arr[:-lag])) for lag in lags]
            
            # Proteção contra erros matemáticos (ex: série flat gerando log(0))
            with np.errstate(divide='ignore', invalid='ignore'):
                try:
                    poly = np.polyfit(np.log(lags), np.log(tau), 1)
                    return float(poly[0] * 2.0)
                except:
                    return float('nan')

        # 4. Aplicação Deslizante (Rolling Map)
        if calc_type == 'simple':
            hurst_series = s.rolling_map(simple_hurst, window_size=window)
        elif calc_type == 'rs':
            hurst_series = s.rolling_map(hurst_rs, window_size=window)
        else:
            raise ValueError(f"Tipo não suportado: {calc_type}")

        return hurst_series.fill_null(0.0).alias(self.name)


