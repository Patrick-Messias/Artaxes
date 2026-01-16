import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
import warnings, sys
warnings.filterwarnings("ignore")
sys.path.append(f'C:\\Users\\Patrick\\Desktop\\ART_Backtesting_Platform\\Backend')
from Indicator import Indicator


class CointegrationIndicator(Indicator):
    """
    Indicador de cointegração entre dois ativos.
    Assume que df já contém os dados de ambos os ativos.
    """

    def __init__(self, asset=None, timeframe=None, **params):
        # Parâmetros específicos para cointegração
        default_params = {
            'window': 252,
            'column_x': 'close',  # Coluna para o primeiro ativo
            'column_y': 'close',  # Coluna para o segundo ativo  
            'scale_output': True
        }
        default_params.update(params)
        super().__init__(asset, timeframe, **default_params)

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcula cointegração entre duas colunas do mesmo DataFrame.
        DataFrame deve conter colunas para ambos os ativos.
        """
        window = self.params.get('window', 252)
        column_x = self.params.get('column_x', 'close')
        column_y = self.params.get('column_y', 'close')
        scale_output = self.params.get('scale_output', True)

        # Verifica se as colunas existem
        if column_x not in df.columns or column_y not in df.columns:
            raise ValueError(f"Colunas {column_x} ou {column_y} não encontradas no DataFrame")

        # Remove NaN values
        valid_data = df[[column_x, column_y]].dropna()
        
        if len(valid_data) < window:
            raise ValueError(f"Dados insuficientes: {len(valid_data)} < {window}")

        coint_tstat = np.full(len(df), np.nan)
        coint_pval = np.full(len(df), np.nan)

        for i in range(window, len(valid_data)):
            try:
                sub_x = valid_data[column_x].iloc[i-window:i]
                sub_y = valid_data[column_y].iloc[i-window:i]
                
                # Teste de cointegração
                tstat, pval, _ = coint(sub_x, sub_y)
                coint_tstat[i] = tstat
                coint_pval[i] = pval
            except Exception as e:
                # Continua em caso de erro no cálculo
                continue

        # Normalização
        if scale_output:
            mean_val = np.nanmean(coint_tstat)
            std_val = np.nanstd(coint_tstat)
            if std_val > 1e-8:
                coint_tstat = (coint_tstat - mean_val) / std_val
                coint_tstat = np.clip(coint_tstat, -3, 3)  # Limita outliers

        return pd.Series(coint_tstat, index=df.index, name='cointegration_tstat')
    

    
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     from statsmodels.tsa.stattools import coint
#     import numpy as np
#     import pandas as pd

#     np.random.seed(42)

#     dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
#     # Cria um ÚNICO DataFrame com ambos os ativos
#     df = pd.DataFrame({
#         'datetime': dates,
#         'xau_ret': np.random.normal(0, 0.01, 500).cumsum(),
#         'usdx_ret': np.random.normal(0, 0.009, 500).cumsum()
#     })

#     # Cria o indicador
#     indicator = CointegrationIndicator(
#         column_x='xau_ret', 
#         column_y='usdx_ret', 
#         window=252, 
#         scale_output=True
#     )
    
#     # Calcula (agora recebe apenas um DataFrame)
#     coint_series = indicator.calculate(df)

#     # Cria DataFrame para plotagem (junta com datetime)
#     df_coint = pd.DataFrame({
#         'datetime': df['datetime'],
#         'cointegration_tstat': coint_series
#     })

#     print("Últimos 10 valores:")
#     print(df_coint.tail(10))

#     # Plotagem
#     plt.figure(figsize=(12, 6))
#     plt.style.use('dark_background')
#     plt.plot(df_coint['datetime'], df_coint['cointegration_tstat'], 
#              label='Cointegration t-stat (normalizado)', linewidth=1, color='cyan')
#     plt.axhline(y=0, color='white', linestyle='--', alpha=0.5)
#     plt.title('Cointegração entre XAU e USDX')
#     plt.ylabel('t-stat (normalizado)')
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()