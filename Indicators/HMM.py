import warnings, sys
warnings.filterwarnings("ignore")
sys.path.append(f'C:\\Users\\Patrick\\Desktop\\ART_Backtesting_Platform\\Backend')

import numpy as np, pandas as pd
from typing import Optional, Callable, Literal, List, Union
from tqdm import tqdm
from copy import deepcopy
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from GARCH import GARCH  
from HawkesProcess import HawkesProcess
from ParkinsonVol import ParkinsonVolatility
from CandlestickPlot import CandlestickPlot
from LewellenBarndorffJump import LewellenBarndorffJump
from HARComponents import HARComponents

class HMM:
    """
    Hidden Markov Model (HMM) para detecção de regimes de volatilidade.

    Características:
    - Usa features como rv, rolling std, prediction (opcional) etc.
    - Determina automaticamente qual estado é high/low volatility
      baseado na média da feature escolhida em 'predict_feature'.
    - Suporta modo rolling ou anchored.
    """

    def __init__(
        self,
        asset: Optional[str] = None,
        timeframe: Optional[str] = None,
        n_states: int = 2,
        covariance_type: Literal['full', 'diag', 'spherical', 'tied'] = 'full',
        n_iter: int = 100,
        tol: float = 1e-4,
        window: Union[int, float] = 252,
        rolling_mode: Literal['rolling', 'anchored'] = 'rolling',
        features: Optional[List[str]] = None,
        predict_feature: str = 'rv',  # <-- NOVO: feature usada para definir high/low vol
        scale_features: bool = True,
        random_state: int = 42,
        verbose: bool = True
    ):
        self.asset = asset
        self.timeframe = timeframe
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.window = window
        self.rolling_mode = rolling_mode
        self.features = features
        self.predict_feature = predict_feature
        self.scale_features = scale_features
        self.random_state = random_state
        self.verbose = verbose

    # =========================================
    # Preparar features
    # =========================================
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Seleciona apenas as colunas que existem no df
        cols = [c for c in self.features if c in df.columns]
        if not cols:
            raise ValueError(f"Nenhuma das features especificadas existe no DataFrame: {self.features}")

        # Cria um DataFrame apenas com as features válidas
        df_feat = df[cols].copy().replace([np.inf, -np.inf], np.nan).fillna(0)

        return df_feat


    # =========================================
    # Escalonar features
    # =========================================
    def _scale(self, X: np.ndarray) -> (np.ndarray, Optional[StandardScaler]):
        if not self.scale_features:
            return X, None
        scaler = StandardScaler()
        return scaler.fit_transform(X), scaler

    # =========================================
    # Mapeia qual estado é high ou low volatility
    # =========================================
    def _map_states_by_vol(self, df_feat: pd.DataFrame, states: np.ndarray) -> np.ndarray:
        """
        Define qual estado é high vol (1) e low vol (0)
        com base na média da coluna escolhida em self.predict_feature.
        """
        if self.predict_feature not in df_feat.columns:
            # fallback se a feature escolhida não existir
            metric = df_feat.iloc[:, 0]
            if self.verbose:
                print(f"Aviso: '{self.predict_feature}' não encontrada, usando {df_feat.columns[0]}.")
        else:
            metric = df_feat[self.predict_feature]

        means = {s: metric[states == s].mean() for s in np.unique(states)}
        ordered = sorted(means.items(), key=lambda x: x[1])  # do menor para o maior
        mapping = {old: new for new, (old, _) in enumerate(ordered)}
        return np.vectorize(lambda x: mapping.get(x, x))(states).astype(int)

    # =========================================
    # Função principal
    # =========================================
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        self.df = df.copy().reset_index(drop=True)

        if 'ret' not in self.df.columns:
            raise ValueError("DataFrame precisa conter a coluna 'ret' (retornos).")

        n = len(self.df)
        self.df['state'] = np.nan

        # Definir janela
        if 0 < self.window < 1:
            split_idx = int(n * self.window)
            test_idx = range(split_idx, n)
        else:
            split_idx = int(self.window)
            test_idx = range(split_idx, n)

        # Caso window == 1 (treina uma vez com todo o histórico)
        if self.window == 1 or split_idx >= n:
            df_feat = self._prepare_features(self.df).fillna(0)
            X = df_feat.values
            Xs, _ = self._scale(X)
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                tol=self.tol,
                random_state=self.random_state
            )
            model.fit(Xs)
            states = model.predict(Xs)
            self.df['state'] = pd.Series(states, index=self.df.index).ffill()#.bfill()

            if self.verbose:
                print("HMM treinado com todos os dados (window==1).")
            return self.df

        # Rolling / Anchored
        for i in tqdm(test_idx, desc="HMM rolling fit"):
            if self.rolling_mode == 'rolling':
                start = max(0, i - split_idx)
                train_df = self.df.iloc[start:i].copy()
            else:
                train_df = self.df.iloc[:i].copy()

            df_feat = self._prepare_features(train_df).fillna(0)
            X = df_feat.values

            if len(X) < max(10, self.n_states * 3):
                self.df.at[i, 'state'] = np.nan
                continue

            try:
                Xs, scaler = self._scale(X)
                model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    tol=self.tol,
                    random_state=self.random_state
                )
                model.fit(Xs)

                feat_i = self._prepare_features(self.df.iloc[[i]]).fillna(0).values
                if self.scale_features and scaler is not None:
                    feat_i = scaler.transform(feat_i)

                state_i = model.predict(feat_i)[0]

                self.df.at[i, 'state'] = int(state_i)
            except Exception as e:
                if self.verbose:
                    print(f"Falha no HMM no índice {i}: {e}")
                self.df.at[i, 'state'] = np.nan
        self.df['state'] = self.df['state'].ffill()#.bfill()

        return self.df
    




def generate_hmm_multi_horizon_features(df, horizons=[21, 63, 126, 252], price_col='close'):
    """
    Gera múltiplas features baseadas em HMM para diferentes janelas (multi-horizon),
    recalculando os indicadores para cada horizonte.
    """
    df_out = df.copy()

    for h in tqdm(horizons, desc="Gerando HMMs multi-horizonte"):
        try:
            # =====================================
            # 1️⃣ Recalcular features para este horizonte
            # =====================================
            temp = df_out.copy()
            temp['rv'] = np.log(temp['high'] / temp['low']).replace([np.inf, -np.inf], np.nan).fillna(0)
            temp['rv_roll_std'] = temp['rv'].rolling(h).std().fillna(0)

            temp['ret'] = np.log(temp[price_col] / temp[price_col].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0)
            temp['ret_roll_std'] = temp['ret'].rolling(h).std().fillna(0)

            temp['hilo'] = np.log(temp['high'].rolling(h).max() / temp['low'].rolling(h).max()).replace([np.inf, -np.inf], np.nan).fillna(0)
            temp['hawkes'] = HawkesProcess(price_col='rv', kappa=0.1).calculate(temp)
            temp['parkinson'] = ParkinsonVolatility(window=h).calculate(temp)

            # =====================================
            # 2️⃣ Rodar HMM para este horizonte
            # =====================================
            hmm = HMM(
                n_states=2,
                window=h,
                rolling_mode='rolling',
                predict_feature='rv',
                features=['rv', 'rv_roll_std', 'ret_roll_std', 'hilo', 'hawkes', 'parkinson'],
                n_iter=42,
                verbose=False
            )

            hmm_df = hmm.calculate(temp)

            # =====================================
            # 3️⃣ Adicionar colunas resultantes ao df principal
            # =====================================
            df_out[f'hmm_state_{h}'] = hmm_df['predicted_state']
            df_out[f'hmm_prob0_{h}'] = hmm_df['state_prob_0']
            df_out[f'hmm_prob1_{h}'] = hmm_df['state_prob_1']

        except Exception as e:
            print(f"Falha ao gerar HMM com janela {h}: {e}")

    return df_out


# =========================================
# Exemplo de uso
# =========================================
if __name__ == "__main__":
    df = pd.read_excel(r'C:\Users\Patrick\Desktop\Artaxes Portfolio\MAIN\MT5_Dados\WIN$_D1.xlsx')
    #df = df[200:]
    df = df[-1000:]

    length = 21*12
    ind_length = 21*12

    # rodar GARCH (opcional)
    try:
        garch = GARCH(window=length, mean_model='Zero', vol_model='Garch', dist='normal')
        df = garch.calculate(df)
    except Exception as e:
        print("Falha ao rodar GARCH:", e)

    # Features Prep
    if 'rv' not in df.columns: df['rv'] = np.log(df['high'] / df['low']).replace([np.inf, -np.inf], np.nan).fillna(0)
    # if 'rv_roll_std' not in df.columns: df['rv_roll_std'] = df['rv'].rolling(ind_length).std().fillna(0)
    # if 'ret' not in df.columns: df['ret'] = np.log(df['close'] / df['close'].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0)
    # if 'ret_roll_std' not in df.columns: df['ret_roll_std'] = df['ret'].rolling(ind_length).std().fillna(0)
    # df['hilo'] = np.log(df['high'].rolling(ind_length).max() / df['low'].rolling(ind_length).max()).replace([np.inf, -np.inf], np.nan).fillna(0)
    df['jump'] = LewellenBarndorffJump(window=ind_length).calculate(df)
    df['hawkes'] = HawkesProcess(price_col='rv', kappa=0.1).calculate(df)
    df['parkinson'] = ParkinsonVolatility(window=ind_length).calculate(df)
    df[['har_daily', 'har_weekly', 'har_monthly']] = HARComponents().calculate(df)
    #df['highest'] = df['high'] >= df['high'].rolling(length).max().shift(1).fillna(False)
    #df['lowest'] = df['low'] <= df['low'].rolling(length).min().shift(1).fillna(False)

    # rodar HMM
    hmm = HMM(
        n_states=3,
        window=length,
        rolling_mode='rolling',
        predict_feature='rv',    # <-- define qual feature representa volatilidade
        features=['rv', 'rv_roll_std', 'ret_roll_std', 'prediction', 'hilo', 'jump', 'hawkes', 'parkinson', 'har_daily', 'har_weekly', 'har_monthly', 'highest', 'lowest'], 
        n_iter=42, 
        verbose=True
    )

    df_hmm = hmm.calculate(df)
    print(df_hmm.head(5))
    print(df_hmm.tail(5))

    if 'datetime' not in df_hmm.columns:
        if 'date' in df_hmm.columns and 'time' in df_hmm.columns:
            df_hmm['datetime'] = pd.to_datetime(df_hmm['date'].astype(str) + ' ' + df_hmm['time'].astype(str))

    # Supondo df_hmm já carregado
    df_hmm['datetime'] = pd.to_datetime(df_hmm['datetime'])
    df_hmm = df_hmm.sort_values('datetime').reset_index(drop=True)
    df_hmm['position'] = df_hmm['state'].ffill()

    # Criar gráfico
    chart = CandlestickPlot(
        df=df_hmm,
        timeframe='D1',
        datetime_col='datetime',
        areas_config={'position': {'data': df_hmm['position'], 'colors':{0: 'green', 1: 'yellow', 2: 'red'}, 'alpha': 0.2, 'label': None}},
        indicators={
            'Jump': {'data': df['jump'], 'color': 'yellow', 'linewidth': 0.8, 'plot_on_graph': False},
            #'HAR D': {'data': df['har_daily'], 'color': 'red', 'linewidth': 0.8, 'plot_on_graph': False},
            #'HAR M': {'data': df['har_monthly'], 'color': 'darkred', 'linewidth': 0.8, 'plot_on_graph': False},
            #'Hawkes': {'data': df['hawkes'], 'color': 'white', 'linewidth': 0.8, 'plot_on_graph': False},
            #'Parkinson': {'data': df['parkinson'], 'color': 'white', 'linewidth': 0.8, 'plot_on_graph': False}
            },
        figsize=(12, 6)
    )
    chart.show(title='', legend=True)

