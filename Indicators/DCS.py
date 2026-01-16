import pandas as pd, numpy as np

import warnings, sys
warnings.filterwarnings("ignore")
sys.path.append(f'C:\\Users\\Patrick\\Desktop\\ART_Backtesting_Platform\\Backend')
from Indicator import Indicator

class DCSIndicator(Indicator):
    """Dynamic Conditional Score (DCS) para volatilidade e skew dinâmicos"""

    def __init__(self, asset=None, timeframe=None, **params):
        # ✅ SIMPLES ASSIM - sem default_params
        super().__init__(asset, timeframe, **params)

    def calculate(self, df: pd.DataFrame):
        window = self.params.get('window', 252)
        scale_output = self.params.get('scale_output', True)

        df = df.copy()

        if len(df) < window:
            raise ValueError(f"Not enough data with window size {window}")

        if 'ret' not in df.columns:
            df['ret'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)

        dcs_vol = np.full(len(df), np.nan)
        dcs_skew = np.full(len(df), np.nan)

        for i in range(window, len(df)):
            sub = df['ret'].iloc[i - window:i]
            
            if len(sub) < 2:
                continue

            # ===== DCS Volatility =====
            mu = sub.mean()
            sigma = sub.std(ddof=0)
            if sigma == 0: 
                sigma = 1e-8
            score_vol = (sub - mu) / sigma**2
            dcs_vol[i] = score_vol.mean()

            # ===== DCS Skew =====
            skew_score = ((sub - mu)**3) / sigma**3
            dcs_skew[i] = skew_score.mean()

        # Escalar output
        if scale_output:
            for arr in [dcs_vol, dcs_skew]:
                non_nan = arr[~np.isnan(arr)]
                if len(non_nan) > 0:
                    mean = np.mean(non_nan)
                    std = np.std(non_nan) + 1e-8
                    arr[~np.isnan(arr)] = np.clip((non_nan - mean) / std, -3, 3)

        return pd.DataFrame({
            'dcs_vol': dcs_vol,
            'dcs_skew': dcs_skew
        }, index=df.index)



# if __name__ == "__main__":
#     df = pd.read_excel(r'C:\Users\Patrick\Desktop\Artaxes Portfolio\MAIN\MT5_Dados\WIN$_D1.xlsx')

#     # Teste 1: Uso básico
#     print("\n" + "="*50)
#     print("TESTE 1: Uso básico")
#     print("="*50)
    
#     dcs = DCSIndicator(window=63, scale_output=True)
#     result = dcs.calculate(df)
    
#     print(f"Resultado shape: {result.shape}")
#     print(f"Colunas: {result.columns.tolist()}")
#     print(f"Primeiros valores não-NaN:")
#     first_valid = result.dropna().head()
#     print(first_valid)

#     # Teste 2: Múltiplos parâmetros com calculate_all_sets
#     print("\n" + "="*50)
#     print("TESTE 2: Múltiplos parâmetros")
#     print("="*50)
    
#     dcs_multi = DCSIndicator(
#         window=[63, 126], 
#         scale_output=[True, False]
#     )
    
#     results = dcs_multi.calculate_all_sets(df)
    
#     print(f"Número de combinações: {len(results['dcsindicator'])}")
#     for param_set, result_df in results['dcsindicator'].items():
#         print(f"  {param_set}: {result_df.shape} - NaNs: {result_df.isna().sum().sum()}")

#     # Teste 3: Verificar estatísticas
#     print("\n" + "="*50)
#     print("TESTE 3: Estatísticas dos resultados")
#     print("="*50)
    
#     for col in ['dcs_vol', 'dcs_skew']:
#         series = result[col].dropna()
#         print(f"{col}:")
#         print(f"  Média: {series.mean():.6f}")
#         print(f"  Std:   {series.std():.6f}")
#         print(f"  Min:   {series.min():.6f}")
#         print(f"  Max:   {series.max():.6f}")
#         print(f"  Não-NaN: {len(series)}")

#     # Plotagem
#     print("\n" + "="*50)
#     print("GERANDO GRÁFICOS...")
#     print("="*50)
    
#     import matplotlib.pyplot as plt
    
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
#     # DCS Volatility
#     ax1.plot(df['date'], result['dcs_vol'], linewidth=1, color='cyan', label='DCS Volatility')
#     ax1.axhline(y=0, color='white', linestyle='--', alpha=0.5)
#     ax1.set_title('DCS Volatility')
#     ax1.grid(True, alpha=0.3)
#     ax1.legend()
    
#     # DCS Skew
#     ax2.plot(df['date'], result['dcs_skew'], linewidth=1, color='magenta', label='DCS Skew')
#     ax2.axhline(y=0, color='white', linestyle='--', alpha=0.5)
#     ax2.set_title('DCS Skew')
#     ax2.grid(True, alpha=0.3)
#     ax2.legend()
    
#     plt.tight_layout()
#     plt.show()

#     print("✅ Todos os testes executados com sucesso!")



