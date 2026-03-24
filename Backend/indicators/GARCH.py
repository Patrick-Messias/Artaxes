# import pandas as pd, numpy as np
# from arch import arch_model
# from typing import Optional, Callable, Literal
# from tqdm import tqdm
# import warnings, sys
# warnings.filterwarnings("ignore")
# sys.path.append(f'C:\\Users\\Patrick\\Desktop\\ART_Backtesting_Platform\\Backend')
# from Indicator import Indicator

# # =========================
# # Defaults para cada modelo
# # =========================
# MEAN_DEFAULTS = {
#     'Zero': None,
#     'Constant': None,
#     'AR': 1,
#     'ARX': 1,
#     'HAR': [1, 5, 22]
# }

# VOL_DEFAULTS = {
#     'Arch': {'p': 1, 'q': 0},
#     'Garch': {'p': 1, 'q': 1},
#     'Figarch': {'p': 1, 'q': 1},
#     'GJR': {'p': 1, 'o': 1, 'q': 1}
# }

# class GARCH(Indicator):
#     def __init__(
#         self,
#         asset: str = None,
#         timeframe: str = None,
#         window: float = 252,
#         mean_model: Literal['Zero', 'Constant', 'AR', 'ARX', 'HAR'] = 'Zero',
#         vol_model: Literal['Arch', 'Garch', 'Figarch', 'GJR'] = 'Garch',
#         dist: Literal['normal', 'gaussian', 'ged', 'generalized error', 'skewstudent', 'skewt', 'studentst', 't'] = 'normal',
#         rolling_mode: Literal['rolling', 'anchored'] = 'rolling',
#         price_col_func: Optional[Callable[[pd.DataFrame], pd.Series]] = lambda df_: np.log(df_['high'] / df_['low']).replace([np.inf, -np.inf], np.nan).ffill(), 
#         verbose: bool = True
#     ):
#         self.asset = asset
#         self.timeframe = timeframe
#         self.window = window
#         self.mean_model = mean_model
#         self.vol_model = vol_model
#         self.dist = dist
#         self.rolling_mode = rolling_mode
#         self.price_col_func = price_col_func
#         self.verbose = verbose

#     # =========================
#     # Função principal de ajuste e previsão
#     # =========================
#     def calculate(self, df: pd.DataFrame):
#         self.df = df.copy().reset_index(drop=True)

#         if len(df) < self.window:
#             raise ValueError(f"Not enough data with window size")

#         # calcular returns
#         self.df['ret'] = self.price_col_func(self.df)

#         if self.window == 1.0:
#             train = self.df['ret'].dropna()
#             vol_type = self.vol_model if self.vol_model in ['Arch', 'Figarch'] else 'GARCH'
#             arch_kwargs = VOL_DEFAULTS.get(self.vol_model, {})
#             am = arch_model(train, mean=self.mean_model, vol=vol_type, dist=self.dist, **arch_kwargs)
#             res = am.fit(disp="off")
#             fc = res.forecast(horizon=1)
#             next_forecast = np.sqrt(fc.variance.values[-1, 0])
#             if self.verbose:
#                 print(f"Forecast próximo período: {next_forecast:.6f}")
#             return next_forecast
        
#         self.df['rv'] = np.log(self.df['high'] / self.df['low']).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
#         self.df.dropna(inplace=True)

#         n = len(self.df)
#         self.predictions = [np.nan] * n

#         # Definir índices de treino/teste
#         if 0 < self.window < 1:
#             split_idx = int(n * self.window)
#             test_idx = range(split_idx, n)
#         else:
#             split_idx = int(self.window)
#             test_idx = range(split_idx, n)

#         # =========================
#         # Treinamento e previsão rolling
#         # =========================
#         for i in tqdm(test_idx):
#             train = self.df['ret'][max(0, i - split_idx):i] if self.rolling_mode == 'rolling' else self.df['ret'][:i]
            
#             try:
#                 arch_kwargs = VOL_DEFAULTS.get(self.vol_model, {})
#                 if self.mean_model in MEAN_DEFAULTS:
#                     arch_kwargs['lags'] = MEAN_DEFAULTS[self.mean_model]

#                 vol_type = self.vol_model if self.vol_model in ['Arch', 'Figarch'] else 'GARCH'
#                 am = arch_model(train, mean=self.mean_model, vol=vol_type, dist=self.dist, **arch_kwargs)
#                 res = am.fit(disp="off")
#                 fc = res.forecast(horizon=1)
#                 pred = np.sqrt(fc.variance.values[-1, 0])
#                 self.predictions[i] = pred if not np.isnan(pred) else train.std()
#             except Exception:
#                 self.predictions[i] = train.std()

#         self.df['prediction'] = self.predictions

#         # =========================
#         # Métricas simples
#         # =========================
#         def _rmse(y_true, y_pred):
#             mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
#             return np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2))

#         def _mae(y_true, y_pred):
#             mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
#             return np.mean(np.abs(y_true[mask] - y_pred[mask]))

#         valid_mask = ~self.df['prediction'].isna()
#         mae = _mae(self.df.loc[valid_mask, 'rv'], self.df.loc[valid_mask, 'prediction'])
#         rmse = _rmse(self.df.loc[valid_mask, 'rv'], self.df.loc[valid_mask, 'prediction'])

#         if self.verbose:
#             print(f"MAE: {mae:.6f}, RMSE: {rmse:.6f}")

#         return self.df

import pandas as pd
import numpy as np
from arch import arch_model
from tqdm import tqdm
import warnings
import sys
warnings.filterwarnings("ignore")
sys.path.append(f'C:\\Users\\Patrick\\Desktop\\ART_Backtesting_Platform\\Backend')
from Indicator import Indicator

class GARCH(Indicator):
    def __init__(self, asset=None, timeframe=None, **params):
        super().__init__(asset, timeframe, **params)

    def calculate(self, df: pd.DataFrame):
        # Obter parâmetros
        window = self.params.get('window', 252)
        mean_model = self.params.get('mean_model', 'Zero')
        vol_model = self.params.get('vol_model', 'Garch')
        dist = self.params.get('dist', 'normal')
        rolling_mode = self.params.get('rolling_mode', 'rolling')
        verbose = self.params.get('verbose', True)
        
        # Defaults para cada modelo
        MEAN_DEFAULTS = {
            'Zero': None,
            'Constant': None,
            'AR': 1,
            'ARX': 1,
            'HAR': [1, 5, 22]
        }

        VOL_DEFAULTS = {
            'Arch': {'p': 1, 'q': 0},
            'Garch': {'p': 1, 'q': 1},
            'Figarch': {'p': 1, 'q': 1},
            'GJR': {'p': 1, 'o': 1, 'q': 1}
        }

        df = df.copy().reset_index(drop=True)

        if len(df) < window:
            raise ValueError(f"Not enough data with window size {window}")

        # Calcular returns
        price_col_func = self.params.get('price_col_func', 
            lambda df_: np.log(df_['high'] / df_['low']).replace([np.inf, -np.inf], np.nan).ffill())
        df['ret'] = price_col_func(df)

        if window == 1.0:
            train = df['ret'].dropna()
            vol_type = vol_model if vol_model in ['Arch', 'Figarch'] else 'GARCH'
            arch_kwargs = VOL_DEFAULTS.get(vol_model, {})
            am = arch_model(train, mean=mean_model, vol=vol_type, dist=dist, **arch_kwargs)
            res = am.fit(disp="off")
            fc = res.forecast(horizon=1)
            next_forecast = np.sqrt(fc.variance.values[-1, 0])
            if verbose:
                print(f"Forecast próximo período: {next_forecast:.6f}")
            return pd.Series([next_forecast] * len(df), index=df.index, name='garch_forecast')

        df['rv'] = np.log(df['high'] / df['low']).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        df.dropna(inplace=True)

        n = len(df)
        predictions = [np.nan] * n

        # Definir índices de treino/teste
        if 0 < window < 1:
            split_idx = int(n * window)
            test_idx = range(split_idx, n)
        else:
            split_idx = int(window)
            test_idx = range(split_idx, n)

        # Treinamento e previsão rolling
        for i in tqdm(test_idx) if verbose else test_idx:
            train = df['ret'][max(0, i - split_idx):i] if rolling_mode == 'rolling' else df['ret'][:i]
            
            try:
                arch_kwargs = VOL_DEFAULTS.get(vol_model, {})
                if mean_model in MEAN_DEFAULTS:
                    arch_kwargs['lags'] = MEAN_DEFAULTS[mean_model]

                vol_type = vol_model if vol_model in ['Arch', 'Figarch'] else 'GARCH'
                am = arch_model(train, mean=mean_model, vol=vol_type, dist=dist, **arch_kwargs)
                res = am.fit(disp="off")
                fc = res.forecast(horizon=1)
                pred = np.sqrt(fc.variance.values[-1, 0])
                predictions[i] = pred if not np.isnan(pred) else train.std()
            except Exception:
                predictions[i] = train.std()

        return pd.Series(predictions, index=df.index, name='garch_forecast')

if __name__ == "__main__":
    # Criar dados de exemplo
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    # Simular dados financeiros realistas
    returns = np.random.normal(0, 0.01, 1000)
    prices = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'datetime': dates,
        'close': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, 1000))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, 1000)))
    })
    
    print("Dados criados:")
    print(df.head())
    print(f"Shape: {df.shape}")

    # Teste 1: Uso básico
    print("\n" + "="*60)
    print("TESTE 1: GARCH Básico")
    print("="*60)
    
    garch = GARCH(
        window=252,
        mean_model='Constant',
        vol_model='Garch', 
        dist='normal',
        verbose=False  # Desativa progress bar para teste
    )
    
    result = garch.calculate(df)
    print(f"Resultado shape: {result.shape}")
    print(f"Primeiros valores válidos:")
    valid_results = result.dropna().head(10)
    print(valid_results)

    # Teste 2: Múltiplos parâmetros
    print("\n" + "="*60)
    print("TESTE 2: Múltiplos Parâmetros")
    print("="*60)
    
    garch_multi = GARCH(
        window=[126, 252],
        vol_model=['Garch', 'GJR'],
        dist=['normal', 'studentst']
    )
    
    results = garch_multi.calculate_all_sets(df)
    
    print(f"Número de combinações: {len(results['garch'])}")
    for param_set, result_series in results['garch'].items():
        valid_data = result_series.dropna()
        print(f"  {param_set}: {len(valid_data)} valores válidos")

    # Teste 3: Diferentes configurações
    print("\n" + "="*60)
    print("TESTE 3: Configurações Específicas")
    print("="*60)
    
    # GARCH com janela menor para teste rápido
    garch_fast = GARCH(
        window=63,
        mean_model='Zero', 
        vol_model='Arch',
        verbose=False
    )
    
    fast_result = garch_fast.calculate(df)
    print(f"GARCH rápido - valores válidos: {len(fast_result.dropna())}")

    # Teste 4: Estatísticas
    print("\n" + "="*60)
    print("TESTE 4: Estatísticas dos Resultados")
    print("="*60)
    
    for param_set, result_series in results['garch'].items():
        valid_data = result_series.dropna()
        if len(valid_data) > 0:
            print(f"\n{param_set}:")
            print(f"  Média: {valid_data.mean():.6f}")
            print(f"  Std:   {valid_data.std():.6f}")
            print(f"  Min:   {valid_data.min():.6f}")
            print(f"  Max:   {valid_data.max():.6f}")

    # Plotagem
    print("\n" + "="*60)
    print("GERANDO GRÁFICOS...")
    print("="*60)
    
    import matplotlib.pyplot as plt
    
    # Pegar o primeiro resultado para plotar
    first_result = list(results['garch'].values())[0]
    
    plt.figure(figsize=(12, 8))
    
    # Preços
    plt.subplot(2, 1, 1)
    plt.plot(df['datetime'], df['close'], linewidth=1, color='white', alpha=0.7, label='Preço')
    plt.title('Preços')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Volatilidade GARCH
    plt.subplot(2, 1, 2)
    plt.plot(df['datetime'], first_result, linewidth=1, color='cyan', label='Volatilidade GARCH')
    plt.title('Previsão de Volatilidade GARCH')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    print("✅ Todos os testes executados com sucesso!")

