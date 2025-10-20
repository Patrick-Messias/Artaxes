import numpy as np
import pandas as pd
from arch import arch_model
from hmmlearn.hmm import GaussianHMM
import statsmodels.api as sm
from pykalman import KalmanFilter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from pprint import pprint
from VolFeatureEng import VolFeatureEng
from VolBinaryPredictor import VolBinaryPredictor
from Indicator import Indicator

# ======================================================
#  VolModelManager
# ======================================================
class VolModelManager:
    """
    Manager for volatility models using features from VolFeatureEng.
    Supports:
        - HAR-RV (linear)
        - GARCH (via arch package)
        - SV (via state-space Kalman filter)
        - HMM (GaussianHMM)
        - Particle Filter placeholder
    """

    def __init__(self, df_features: pd.DataFrame):
        self.df = df_features.copy()
        self.models = {}

    # --------------------------------------------------
    # 1. HAR-RV model
    def fit_har_rv(self, base_col='vol_yz'):
        """
        Ajusta o modelo HAR-RV (Heterogeneous Autoregressive Realized Volatility)
        e adiciona a previsão ao DataFrame como 'har_pred'.

        base_col: str
            Nome da coluna base de volatilidade (ex: 'vol_yz').
        """
        # Define a variável alvo (volatilidade do próximo período)
        y = self.df[base_col].shift(-1)

        # Define as features do modelo
        X = self.df[[f'{base_col}_daily', f'{base_col}_weekly', f'{base_col}_monthly']]

        # Remove linhas com NaN simultaneamente em X e y
        valid_idx = X.dropna().index.intersection(y.dropna().index)
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        # Ajusta o modelo HAR-RV
        model = LinearRegression().fit(X, y)
        self.models['HAR'] = model

        # Cria vetor de previsões alinhado ao DataFrame original
        pred = pd.Series(index=self.df.index, dtype=float)
        pred.loc[valid_idx] = model.predict(X)

        # Adiciona a previsão ao DataFrame principal
        self.df['har_pred'] = pred

        return model

    # --------------------------------------------------
    # 2. GARCH(1,1) model
    def fit_garch(self, return_col='ret'):
        y = self.df[return_col].dropna() * 100  # arch expects % returns
        model = arch_model(y, mean='Zero', vol='GARCH', p=1, q=1)
        res = model.fit(disp='off')
        self.models['GARCH'] = res
        self.df['garch_vol'] = res.conditional_volatility
        return res

    # --------------------------------------------------
    # 3. Stochastic Volatility (SV) using state-space / Kalman
    def fit_sv(self, return_col='ret'):
        ret = self.df[return_col].dropna()
        log_ret2 = np.log(ret**2 + 1e-8)
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            transition_covariance=np.eye(1) * 0.01,
            observation_covariance=np.eye(1) * 0.1,
        )
        state_means, _ = kf.filter(log_ret2.values.reshape(-1, 1))
        self.df.loc[ret.index, 'sv_logvar'] = state_means.flatten()
        self.df['sv_vol'] = np.exp(0.5 * self.df['sv_logvar'])
        self.models['SV'] = kf
        return kf

    # --------------------------------------------------
    # 4. Hidden Markov Model (volatility regimes)
    def fit_hmm(self, cols=['abs_ret']):
        X = self.df[cols].dropna().values
        model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=200)
        model.fit(X)
        self.df.loc[self.df[cols].dropna().index, 'hmm_state'] = model.predict(X)
        self.models['HMM'] = model
        return model

    # --------------------------------------------------
    # 5. State-space / Particle filter placeholder
    def fit_state_space(self, obs_col='vol_yz'):
        # A very basic linear Gaussian state-space model
        y = self.df[obs_col].dropna()
        model = sm.tsa.UnobservedComponents(y, level='local level')
        res = model.fit(disp=False)
        self.models['StateSpace'] = res
        self.df.loc[y.index, 'ss_level'] = res.level.smoothed
        return res

    # --------------------------------------------------
    # 6. Summary / unified forecast interface
    def forecast_next_vol(self):
        preds = {}
        if 'HAR' in self.models:
            har_cols = ['vol_yz_daily', 'vol_yz_weekly', 'vol_yz_monthly']
            X_last = self.df[har_cols].iloc[[-1]]
            preds['HAR'] = float(self.models['HAR'].predict(X_last))
        if 'GARCH' in self.models:
            preds['GARCH'] = float(self.models['GARCH'].conditional_volatility.iloc[-1])
        if 'SV' in self.models:
            preds['SV'] = float(self.df['sv_vol'].iloc[-1])
        if 'StateSpace' in self.models:
            preds['StateSpace'] = float(self.df['ss_level'].iloc[-1])
        preds['Average'] = np.mean(list(preds.values())) if preds else np.nan
        return preds

    # --------------------------------------------------
    # 7. Summary
    def summary(self):
        print("Available models:", list(self.models.keys()))
        print(self.df.tail()[['har_pred','garch_vol','sv_vol','ss_level','hmm_state']].dropna(how='all'))

    # -------------------------------------------------- #

class VolPredictor(Indicator):
    def __init__(self, asset, timeframe: str, har_period_s: int = 5, har_period_m: int = 21, vol_yz_period_m: int = 21):
        super().__init__(asset, timeframe)
        self.har_period_s = har_period_s
        self.har_period_m = har_period_m
        self.vol_yz_period_m = vol_yz_period_m

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        vfe = VolFeatureEng(df, params={
            'har_period_s': self.har_period_s,
            'har_period_m': self.har_period_m,
            'vol_yz_period_m': self.vol_yz_period_m
        })
        df_feat = vfe.build_all()

        manager = VolModelManager(df_feat)
        manager.fit_har_rv()
        manager.fit_garch()
        manager.fit_sv()
        manager.fit_state_space()
        manager.fit_hmm()

        vbp = VolBinaryPredictor(manager.df, vol_col='vol_yz', period=self.vol_yz_period_m)
        vbp.create_target()
        vbp.select_features()
        vbp.train_model()
        df_with_pred = vbp.predict_all()

        # Alinha com df original pelo índice posicional e preenche NaNs com 0
        vol_up_pred_aligned = pd.Series(0, index=df.index, name='vol_up_pred')
        common_index = min(len(df), len(df_with_pred))
        vol_up_pred_aligned.iloc[:common_index] = df_with_pred['vol_up_pred'].values[:common_index]

        return vol_up_pred_aligned


"""
# Step 1: Build volatility features
df = pd.read_csv(f"c:\\Users\\Patrick\\Desktop\\Artaxes Portfolio\\MAIN\\MT5_Dados\\Forex\\EURUSD_D1.csv")
df = df[:int(len(df)*0.5)]  # limit for testing
print(df)
params={'har_period_s': 5, 'har_period_m': 21, 'vol_yz_period_m': 21}

vfe = VolFeatureEng(df, params=params)      # from OHLC data
df_feat = vfe.build_all()

# Step 2: Feed into unified model manager
manager = VolModelManager(df_feat)

# Fit all desired models
manager.fit_har_rv()
manager.fit_garch()
manager.fit_sv()
manager.fit_state_space()
manager.fit_hmm()

# Step 3: View unified summary and forecast
manager.summary()
pprint(manager.forecast_next_vol())

print(f"\n\n\n")

vbp = VolBinaryPredictor(manager.df, vol_col='vol_yz', period=params['vol_yz_period_m'])
vbp.create_target()
vbp.select_features()
vbp.train_model()
df_with_pred = vbp.predict_all()

print(df_with_pred.value_counts())
"""


