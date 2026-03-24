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
from LinearRegression import LinearRegressionInd
from HURST import HURST

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
    def _scale(self, X: np.ndarray) -> (np.ndarray, Optional[StandardScaler]): # type: ignore
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
        df = df.copy().reset_index(drop=True)

        if len(df) < self.window:
            raise ValueError(f"Not enough data with window size")

        if 'ret' not in df.columns:
            raise ValueError("DataFrame precisa conter a coluna 'ret' (retornos).")

        n = len(df)
        df['state'] = np.nan
        prob_cols = [f'state_prob_{i}' for i in range(self.n_states)]
        for c in prob_cols:
            df[c] = np.nan

        # Definir janela
        if 0 < self.window < 1:
            split_idx = int(n * self.window)
            test_idx = range(split_idx, n)
        else:
            split_idx = int(self.window)
            test_idx = range(split_idx, n)

        # Caso window == 1 (treina uma vez com todo o histórico)
        if self.window == 1 or split_idx >= n:
            df_feat = self._prepare_features(df).fillna(0)
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
            probs = model.predict_proba(Xs)

            # Mapeia high/low vol
            mapped_states = self._map_states_by_vol(df_feat, states)
            df['state'] = mapped_states

            for i in range(self.n_states):
                df[f'state_prob_{i}'] = probs[:, i]

            if self.verbose:
                print("HMM treinado com todos os dados (window==1).")
            return df

        # Rolling / Anchored
        for i in tqdm(test_idx, desc="HMM rolling fit"):
            if self.rolling_mode == 'rolling':
                start = max(0, i - split_idx)
                train_df = df.iloc[start:i].copy()
            else:
                train_df = df.iloc[:i].copy()

            df_feat = self._prepare_features(train_df).fillna(0)
            X = df_feat.values

            if len(X) < max(10, self.n_states * 3):
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

                feat_i = self._prepare_features(df.iloc[[i]]).fillna(0).values
                if self.scale_features and scaler is not None:
                    feat_i = scaler.transform(feat_i)

                # Probabilidades do ponto atual
                probs = model.predict_proba(feat_i)
                last_probs = probs[0, :]

                # Estado mais provável
                state_i = np.argmax(last_probs)
                df.at[i, 'state'] = int(state_i)
                for s in range(self.n_states):
                    df.at[i, f'state_prob_{s}'] = last_probs[s]

            except Exception as e:
                if self.verbose:
                    print(f"Falha no HMM no índice {i}: {e}")
                continue

        # Re-map final dos estados para coerência de volatilidade
        valid_idx = df['state'].dropna().index
        if len(valid_idx) > 0:
            df_feat_valid = self._prepare_features(df.loc[valid_idx]).fillna(0)
            states_valid = df.loc[valid_idx, 'state'].values.astype(int)
            
            # Criar mapping baseado na volatilidade média
            state_vol_means = {}
            for state in np.unique(states_valid):
                state_mask = states_valid == state
                if self.predict_feature in df_feat_valid.columns:
                    state_vol_means[state] = df_feat_valid.loc[valid_idx[state_mask], self.predict_feature].mean()
                else:
                    # Fallback para primeira feature
                    state_vol_means[state] = df_feat_valid.iloc[:, 0][state_mask].mean()
            
            # Ordenar estados por volatilidade (menor para maior)
            sorted_states = sorted(state_vol_means.items(), key=lambda x: x[1])
            state_mapping = {old_state: new_state for new_state, (old_state, _) in enumerate(sorted_states)}
            
            # Aplicar mapping aos estados
            mapped_states = np.array([state_mapping[s] for s in states_valid])
            df.loc[valid_idx, 'state'] = mapped_states
            
            # Reordenar probabilidades de acordo com o mapping
            prob_cols = [f'state_prob_{i}' for i in range(self.n_states)]
            prob_matrix = df.loc[valid_idx, prob_cols].values
            
            # Criar nova matriz de probabilidades com a ordem correta
            new_prob_matrix = np.zeros_like(prob_matrix)
            for old_state, new_state in state_mapping.items():
                new_prob_matrix[:, new_state] = prob_matrix[:, old_state]
            
            df.loc[valid_idx, prob_cols] = new_prob_matrix

        # CORREÇÃO: Preenchimento mais conservador das probabilidades
        prob_cols = [f'state_prob_{i}' for i in range(self.n_states)]
        
        # Primeiro, garanta que as probabilidades somem 1 onde foram calculadas
        for idx in df.index:
            if not df.loc[idx, prob_cols].isna().all():
                prob_sum = df.loc[idx, prob_cols].sum()
                if prob_sum > 0:  # Evita divisão por zero
                    df.loc[idx, prob_cols] = df.loc[idx, prob_cols] / prob_sum
        
        # Apenas forward fill para preencher gaps, sem interpolação agressiva
        for col in prob_cols:
            df[col] = df[col].ffill().bfill()
        
        df['state'] = df['state'].ffill().bfill()

        return df


import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# Importar seus indicadores:
# from Indicators import HMM, HawkesProcess, ParkinsonVolatility, HURST, LewellenBarndorffJump, HARComponents, LinearRegressionInd

def generate_hmm_multi_horizon_features(df, horizons=[21, 63, 126, 252]):
    """
    Gera múltiplas features baseadas em HMM para diferentes janelas (multi-horizon),
    recalculando os indicadores e obtendo um estado de consenso ponderado.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame com colunas ['open', 'high', 'low', 'close'].
    horizons : list[int]
        Lista de janelas (em períodos) para rodar os HMMs.
    price_col : str
        Coluna usada como referência de preço (default: 'close').

    Retorno:
    --------
    df_out : pd.DataFrame
        DataFrame original acrescido das colunas:
            hmm_state_<h>, hmm_prob0_<h>, hmm_prob1_<h> ...
            hmm_prob0_consensus, hmm_prob1_consensus, state_consensus
    """

    df_out = df.copy()

    for h in tqdm(horizons, desc="Gerando HMMs multi-horizonte"):
        try:
            # =====================================
            # 1️⃣ Recalcular features para este horizonte
            # =====================================
            temp = df_out.copy()

            try:
                garch = GARCH(window=h, mean_model='Zero', vol_model='Garch', dist='normal')
                garch_ret = garch.calculate(temp)
            except Exception as e:
                print("Falha ao rodar GARCH:", e)
            temp['garch_prediction'] = garch_ret['prediction']

            temp['rv'] = np.log(temp['high'] / temp['low']).replace([np.inf, -np.inf], np.nan).fillna(0)
            temp['hilo'] = np.log(temp['high'].rolling(h).max() / temp['low'].rolling(h).max()).replace([np.inf, -np.inf], np.nan).fillna(0)
            temp['hawkes'] = HawkesProcess(price_col='rv', kappa=0.1).calculate(temp)
            temp['parkinson'] = ParkinsonVolatility(window=h).calculate(temp)
            temp['hurst'] = HURST(window=h).calculate(temp)
            temp['jump'] = LewellenBarndorffJump(window=h).calculate(temp)
            temp[['har_daily', 'har_weekly', 'har_monthly']] = HARComponents().calculate(temp)
            temp['linear_regression_preds'] = LinearRegressionInd(window=h, target_col='rv').calculate(temp)
            temp['highest'] = temp['high'] >= temp['high'].rolling(h).max().shift(1).fillna(False)
            temp['lowest'] = temp['low'] <= temp['low'].rolling(h).min().shift(1).fillna(False)

            # =====================================
            # 2️⃣ Rodar HMM para este horizonte
            # =====================================
            hmm = HMM(
                n_states=3,
                window=h,
                rolling_mode='rolling',
                predict_feature='rv',
                features=['garch_prediction', 'rv', 'hilo', 'hawkes', 'parkinson', 'hurst', 'jump', 
                          'har_daily', 'har_weekly', 'har_monthly', 'linear_regression_preds', 'highest', 'lowest'],
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
            if 'state_prob_2' in hmm_df.columns:
                df_out[f'hmm_prob2_{h}'] = hmm_df['state_prob_2']

        except Exception as e:
            print(f"⚠️ Falha ao gerar HMM com janela {h}: {e}")

    # =====================================
    # 4️⃣ Estado consenso ponderado entre horizontes
    # =====================================
    prob_cols = [col for col in df_out.columns if 'hmm_prob' in col]
    if prob_cols:
        # Identificar número de estados dinamicamente
        states = sorted({col.split('_')[-2].replace('prob', '') for col in prob_cols if 'prob' in col})
        horizons_found = sorted({int(col.split('_')[-1]) for col in prob_cols})

        # Pesos (menor horizonte => maior peso)
        weights = np.array([1/h for h in horizons_found])
        weights = weights / weights.sum()

        # Média ponderada das probabilidades por estado
        for s in states:
            cols_s = [f'hmm_prob{s}_{h}' for h in horizons_found if f'hmm_prob{s}_{h}' in df_out.columns]
            if cols_s:
                df_out[f'hmm_prob{s}_consensus'] = np.average(df_out[cols_s], axis=1, weights=weights)

        # Estado com probabilidade mais alta (consenso final)
        prob_consensus_cols = [f'hmm_prob{s}_consensus' for s in states if f'hmm_prob{s}_consensus' in df_out.columns]
        df_out['state_consensus'] = (
            df_out[prob_consensus_cols]
            .idxmax(axis=1)
            .str.extract(r'(\d+)')
            .astype(float)
        )

    return df_out

def _smooth_state_transition(df: pd.DataFrame, col: str = 'state', confirm_bars: int = 2) -> pd.Series:
    """
    Exige 'confirm_bars' confirmações consecutivas antes de mudar de estado.
    Ex: se confirm_bars=2, só troca de estado se o novo valor aparecer 2 vezes seguidas.
    """
    states = df[col].values.copy()
    smoothed = [states[0]]
    count = 0  # contador de confirmações

    for i in range(1, len(states)):
        if states[i] != smoothed[-1]:
            count += 1
            # se o novo estado persistir confirm_bars vezes consecutivas, confirma a mudança
            if count >= confirm_bars and all(states[i - confirm_bars + 1:i + 1] == states[i]):
                smoothed.append(states[i])
                count = 0
            else:
                smoothed.append(smoothed[-1])
        else:
            count = 0
            smoothed.append(states[i])
    return pd.Series(smoothed, index=df.index)

# =========================================
# Exemplo de uso
# =========================================
if __name__ == "__main__":
    # sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform')
    # from FRED_get_data import * # type: ignore
    # tickers_basic = {
    #     'SP500': {
    #         'ticker': 'SP500', 
    #         'timeframe': 'daily',
    #         'lag_bdays': 0,
    #         'days_after_month_end': 0
    #     }}
    # df = get_economic_data(tickers_config=tickers_basic) # type: ignore
    # df.columns = df.columns.str.lower()

    df = pd.read_csv(r'C:\Users\Patrick\Desktop\Artaxes Portfolio\MAIN\MT5_Dados\Commodities\XAUUSD_D1.csv')
    df.columns = df.columns.str.lower()
    df = df[-1000:]
    #df = df[-500:]

    length = 252
    ind_length = 252

    # rodar GARCH (opcional)
    try:
        garch = GARCH(window=length, mean_model='Zero', vol_model='Garch', dist='normal')
        df = garch.calculate(df)
    except Exception as e:
        print("Falha ao rodar GARCH:", e)

    # Features Prep
    if 'rv' not in df.columns: df['rv'] = np.log(df['high'] / df['low']).replace([np.inf, -np.inf], np.nan).fillna(0)
    if 'rv_roll_std' not in df.columns: df['rv_roll_std'] = df['rv'].rolling(ind_length).std().fillna(0)
    #if 'ret' not in df.columns: df['ret'] = np.log(df['close'] / df['close'].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0)
    #if 'ret_roll_std' not in df.columns: df['ret_roll_std'] = df['ret'].rolling(ind_length).std().fillna(0)
    df['hurst'] = HURST(window=ind_length, price_col='rv').calculate(df)
    df['hilo'] = np.log(df['high'].rolling(ind_length).max() / df['low'].rolling(ind_length).max()).replace([np.inf, -np.inf], np.nan).fillna(0)
    df['jump'] = LewellenBarndorffJump(window=ind_length).calculate(df)
    df['parkinson'] = ParkinsonVolatility(window=ind_length).calculate(df)
    df['hawkes'] = HawkesProcess(price_col='parkinson', kappa=0.1).calculate(df)
    df[['har_daily', 'har_weekly', 'har_monthly']] = HARComponents().calculate(df)
#    df['linear_regression_preds'] = LinearRegressionInd(window=ind_length, target_col='rv').calculate(df)
#    df['highest'] = df['high'] >= df['high'].rolling(length).max().shift(1).fillna(False)
#    df['lowest'] = df['low'] <= df['low'].rolling(length).min().shift(1).fillna(False)
#    df['vol_avg'] = df['rv'] > df['rv_roll_std']

    # Features to Predict
    # df['rv'] = np.log(df['high'] / df['low']).replace([np.inf, -np.inf], np.nan).fillna(0)
    # mean, std = df['rv'].mean(), df['rv'].std()
    # df['dir_std'] = np.where(df['rv'] > mean + std, 1, np.where(df['rv'] < mean - std, -1, 0))
    # df['dir_mean'] = np.where(df['rv'] > mean, 1, -1)
    # df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    feat = ['rv_roll_std', 'ret_roll_std', 'prediction', 'hilo', 'jump', 'hurst', 'vol_avg',
            'hawkes', 'parkinson', 'har_daily', 'har_weekly', 'har_monthly', 'rv',
            'linear_regression_preds', 'highest', 'lowest', 'arima_prediction'] 

    # rodar HMM
    hmm = HMM(
        n_states=3,
        window=length,
        rolling_mode='rolling',
        predict_feature='rv',    # <-- define qual feature representa volatilidade
        features=feat, 
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
    df_hmm['position'] = _smooth_state_transition(df_hmm, col='position', confirm_bars=2)

    # Criar gráfico
    chart = CandlestickPlot(
        df=df_hmm,
        timeframe='D1',
        datetime_col='datetime',
        areas_config={'position': {'data': df_hmm['position'], 'colors':{0: 'green', 1: 'yellow', 2: 'red'}, 'alpha': 0.15, 'label': None}},
        indicators={
            'rv': {'data': df['rv'], 'color': 'gray', 'linewidth': 0.8, 'plot_on_graph': False},
            #'HAR D': {'data': df['har_daily'], 'color': 'red', 'linewidth': 0.8, 'plot_on_graph': False},
            #'HAR M': {'data': df['har_monthly'], 'color': 'darkred', 'linewidth': 0.8, 'plot_on_graph': False},
            #'Hawkes': {'data': df['hawkes'], 'color': 'white', 'linewidth': 0.8, 'plot_on_graph': False},
            #'Parkinson': {'data': df['parkinson'], 'color': 'white', 'linewidth': 0.8, 'plot_on_graph': False}
            },
        figsize=(12, 6)
    )
    chart.show(title='', legend=True)



    # 1 Correlação com a volatilidade real (ou o estado HMM)
    valid_feat = []
    corrs = {}
    for col in feat:
        if col in df_hmm.columns:
            valid_feat.append(col)
            corrs[col] = df_hmm[col].corr(df_hmm['state'])

    corr_df = pd.DataFrame.from_dict(corrs, orient='index', columns=['corr_with_state'])
    corr_df['abs_corr'] = corr_df['corr_with_state'].abs()
    corr_df = corr_df.sort_values('abs_corr', ascending=False)
    
    print('\nCorrelation with real vol rank','\n',corr_df)



    # 2 Regressão Logística Multinomial
    # ===============================
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    import pandas as pd

    # Preenche valores ausentes
    df_hmm = df_hmm.fillna(0)

    # Seleciona as features válidas e o alvo
    X = df_hmm[valid_feat].copy()
    y = df_hmm['state'].copy()

    # Remove linhas com NaN em X ou y
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    # Remove colunas com desvio padrão zero (sem variação)
    constant_cols = X.columns[X.std() == 0]
    if len(constant_cols) > 0:
        print(f"Removendo colunas constantes: {list(constant_cols)}")
        X = X.drop(columns=constant_cols)

    # Normaliza de forma segura
    X = (X - X.mean()) / (X.std().replace(0, 1))

    # Verificação final
    if X.isna().sum().sum() > 0:
        print("Aviso: ainda há NaNs após normalização, substituindo por 0")
        X = X.fillna(0)

    # Ajuste do modelo multinomial
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model.fit(X, y)

    # Importância média absoluta dos coeficientes por variável
    importance = pd.Series(np.mean(np.abs(model.coef_), axis=0), index=X.columns)
    importance = importance.sort_values(ascending=False)

    print("\nLogistic Regression rank (feature importance):")
    print(importance)

    # (Opcional) Diagnóstico adicional
    print(f"Número de features utilizadas: {X.shape[1]}")






