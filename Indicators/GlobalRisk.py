import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from hmmlearn.hmm import GaussianHMM
import warnings
import sys
warnings.filterwarnings("ignore")
sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend')
sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform')
from Indicator import Indicator
from Asset import Asset
from FRED_get_data import * # type: ignore

class GlobalRisk(Indicator):
    
    # Indicador global de Risk-On / Risk-Off usando PCA + HMM.
    # Retorna:
    #     - RiskIndex: contínuo [-1, 1]
    #     - RiskOn_Prob: probabilidade do regime Risk-On
    #     - RiskOff_Prob: probabilidade do regime Risk-Off
    

    def __init__(
        self,
        assets: dict = None,       # dicionário {'XAUUSD': df_xau, 'SP500': df_sp500, ...}
        timeframe: str = 'D1',
        column_name: str = 'ret',  # coluna de retorno
        window: int = 252,         # janela rolante
        n_components: int = 1,     # componentes principais
        n_states: int = 2          # estados HMM
    ):
        super().__init__(asset=None, timeframe=timeframe)
        self.assets = assets or {}
        self.column_name = column_name
        self.window = window
        self.n_components = n_components
        self.n_states = n_states

    def calculate(self):
        # ==== Preparar DataFrame global ====
        dfs = []
        for name, df in self.assets.items():
            if isinstance(df, Asset):
                df = Asset.data_get(self.timeframe) 
            else:
                df = df.copy()
            if self.column_name not in df.columns:
                df[self.column_name] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
            df = df[['datetime', self.column_name]].rename(columns={self.column_name: name})
            dfs.append(df)

        # Merge por datetime
        df_all = dfs[0]
        for df in dfs[1:]:
            df_all = pd.merge(df_all, df, on='datetime', how='inner')

        df_all.sort_values('datetime', inplace=True)
        df_all.reset_index(drop=True, inplace=True)

        # ==== Inicializar colunas ====
        risk_index = np.full(len(df_all), np.nan)
        risk_on_prob = np.full(len(df_all), np.nan)
        risk_off_prob = np.full(len(df_all), np.nan)

        # ==== Janela rolante ====
        for i in range(self.window, len(df_all)+1):
            sub = df_all.iloc[i-self.window:i, 1:]  # apenas valores de retorno
            # Normalizar
            sub_norm = (sub - sub.mean()) / (sub.std() + 1e-8)

            # PCA
            pca = PCA(n_components=self.n_components)
            factors = pca.fit_transform(sub_norm.values)

            # HMM
            hmm = GaussianHMM(n_components=self.n_states, covariance_type='full', n_iter=500)
            hmm.fit(factors)

            # Probabilidades do último ponto
            probs = hmm.predict_proba(factors)[-1]

            # Determinar qual estado é Risk-On (assume maior média do fator principal)
            means = [factors[hmm.predict(factors)==k, 0].mean() for k in range(self.n_states)]
            risk_on_state = np.argmax(means)

            risk_on_prob[i-1] = probs[risk_on_state]
            risk_off_prob[i-1] = 1 - probs[risk_on_state]
            risk_index[i-1] = 2*probs[risk_on_state]-1  # [-1,1]

        df_result = df_all[['datetime']].copy()
        df_result['risk_index'] = risk_index
        df_result['riskon_prob'] = risk_on_prob
        df_result['riskoff_prob'] = risk_off_prob

        return df_result


# =========================
# EXEMPLO DE USO
# =========================
if __name__ == "__main__":

    tickers_basic = {
        'SP500': {
            'ticker': 'SP500', 
            'timeframe': 'daily',
            'lag_bdays': 0,
            'days_after_month_end': 0
        },
        'VIX': {
            'ticker': 'VIXCLS',
            'timeframe': 'daily', 
            'lag_bdays': 0,
            'days_after_month_end': 0
        },
        'USD_Index': {
            'ticker': 'DTWEXBGS',
            'timeframe': 'daily',
            'lag_bdays': 1, 
            'days_after_month_end': 0
        },
        'US10Y': {
            'ticker': 'DGS10',
            'timeframe': 'daily',
            'lag_bdays': 0,
            'days_after_month_end': 0
        },
        'OIL': {
            'ticker': 'DCOILWTICO',
            'timeframe': 'daily',
            'lag_bdays': 0,
            'days_after_month_end': 0
        },
        # 'COPPER': {
        #     'ticker': 'PCOPPUSDM',
        #     'timeframe': 'monthly',
        #     'lag_bdays': 0,
        #     'days_after_month_end': 0
        # },
        'BTC': {
            'ticker': 'CBBTCUSD',
            'timeframe': 'daily',
            'lag_bdays': 0,
            'days_after_month_end': 0
        },
    }
    
    df_basic = get_economic_data(tickers_config=tickers_basic) # type: ignore
    print(df_basic)

    #df_basic = df_basic[-273:] #df_basic[int(len(df_basic)*0.5):]

    assets = {}
    for col in df_basic.columns:
        assets[col] = pd.DataFrame({
            'datetime': pd.to_datetime(df_basic.index),
            'close': df_basic[col]
        })

    indicator = GlobalRisk(assets=assets, window=252)
    df_risk = indicator.calculate()

    print(df_risk['risk_index']*100)

    # ==== Plot do índice ====
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,5))
    plt.plot(df_risk['datetime'], df_risk['risk_index'], linewidth=1.2, alpha=0.8)
    #plt.fill_between(df_risk['datetime'], df_risk['RiskOn_Prob'], -df_risk['RiskOff_Prob'], color='green', alpha=0.1)
    #plt.title("Global Risk-On / Risk-Off Index")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()




