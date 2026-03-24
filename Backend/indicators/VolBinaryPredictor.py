from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

class VolBinaryPredictor:
    """
    Usa as features de modelos de volatilidade (HAR, GARCH, SV, StateSpace, HMM, etc.)
    para treinar um XGBoost que prevê se a volatilidade do próximo dia será >= média_n.
    """

    def __init__(self, df: pd.DataFrame, vol_col='vol_yz', period=21):
        self.df = df.copy()
        self.vol_col = vol_col
        self.period = period
        self.model = None

    # --------------------------------------------------
    # 1. Cria target binário
    def create_target(self):
        self.df['vol_mean'] = self.df[self.vol_col].rolling(self.period).mean()
        # target = 1 se vol amanhã >= média, 0 caso contrário
        self.df['target'] = (self.df[self.vol_col].shift(-1) >= self.df['vol_mean']).astype(int)
        self.df.dropna(subset=['target'], inplace=True)  # remove última linha sem target
        return self.df

    # --------------------------------------------------
    # 2. Seleciona features
    def select_features(self):
        feature_cols = [
            'har_pred', 'garch_vol', 'sv_vol', 'ss_level', 'hmm_state',
            'vol_parkinson', 'vol_gk', 'jump_abs', 'dow', 'month'
        ]
        # garante que só use colunas presentes no df
        self.features = [c for c in feature_cols if c in self.df.columns]
        return self.features

    # --------------------------------------------------
    # 3. Treina XGBoost
    def train_model(self, test_size=0.2, random_state=42):
        X = self.df[self.features]
        y = self.df['target']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.model.fit(X_train, y_train)

        # avaliação rápida
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"[XGBoost] Accuracy on test set: {acc:.4f}")

        return self.model

    # --------------------------------------------------
    # 4. Gera previsão para todo o DataFrame
    def predict_all(self):
        self.df['vol_up_pred'] = self.model.predict(self.df[self.features])
        return self.df[['vol_up_pred']]


