import pandas as pd
import numpy as np
import warnings, sys
warnings.filterwarnings("ignore")
sys.path.append(f'C:\\Users\\Patrick\\Desktop\\ART_Backtesting_Platform\\Backend')
from Indicator import Indicator

class MOMERSION(Indicator):
    # Calcula o Momersion Indicator (0–100), que mede a proporção de períodos
    # de momentum (continuação) versus mean-reversion (inversão) dentro de
    # uma janela rolante.

    # Fórmula:
    #     Momersion(n) = 100 * MOMc / (MOMc + MRc)
    # onde:
    #     MOMc = contagem de períodos com r(i) * r(i-1) > 0  (mesmo sinal)
    #     MRc  = contagem de períodos com r(i) * r(i-1) < 0  (sinais opostos)

    # Interpretação:
    #     - Mais próximo de 100  → mercado com momentum (tendência forte)
    #     - Momersion ~ 50       → comportamento misto / neutro
    #     - Mais próximo de 0    → mercado com mean reversion (reversões frequentes)
    

    def __init__(self, asset=None, timeframe: str=None, column_name: str='ret', window: int=63):
        super().__init__(asset, timeframe)
        self.column_name = column_name
        self.window = window

    def calculate(self, df: pd.DataFrame):
        if self.column_name not in df.columns:
            if self.column_name == 'ret':
                df['ret'] = df['close'].pct_change().fillna(0)
            else: raise ValueError(f"DataFrame needs column '{self.column_name}'.")

        ret = df[self.column_name].astype(float)
        prod = ret * ret.shift(1)

        # Vetorização: sinais de momentum (+) e mean-reversion (-)
        mom_flag = (prod > 0).astype(int)
        mr_flag = (prod < 0).astype(int)

        # Contagem rolante
        mom_count = mom_flag.rolling(self.window, min_periods=2).sum()
        mr_count = mr_flag.rolling(self.window, min_periods=2).sum()

        momersion = 100 * mom_count / (mom_count + mr_count)
        momersion = momersion.replace([np.inf, -np.inf], np.nan)

        return pd.Series(momersion, index=df.index, name='momersion')



# if __name__ == "__main__":
#     df = pd.read_csv(r'C:\Users\Patrick\Desktop\Artaxes Portfolio\MAIN\MT5_Dados\Commodities\XAUUSD_D1.csv')
#     df.columns = df.columns.str.lower()
#     df = df[-1000:]

#     indicator = MOMERSION(window=21)
#     df['momersion'] = indicator.calculate(df)

#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(12,5))
#     plt.style.use('dark_background')
#     plt.plot(df['datetime'], df['momersion'], color='cyan')
#     plt.axhline(50, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
#     plt.title('Momersion Indicator (Momentum vs Mean Reversion)')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
