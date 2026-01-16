import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
import warnings, sys
warnings.filterwarnings("ignore")
sys.path.append(f'C:\\Users\\Patrick\\Desktop\\ART_Backtesting_Platform\\Backend')
from Indicator import Indicator

class LinearRegressionInd(Indicator):
    def __init__(
        self,
        asset=None,
        timeframe: str = None,
        features: list = None,
        target_col: str = 'ret',
        window: int = 63,
        scale: bool = True,
        add_intercept: bool = True,
        normalize_output: bool = True,
    ):
        
        # Indicador baseado em regressão linear móvel.
        # ----------
        # -> asset : str, opcional           Nome do ativo.
        # -> timeframe : str, opcional       Timeframe da série.
        # -> features : list, opcional       Lista de colunas usadas como features (X). Se None, usa ['ret'].
        # -> target_col : str                Coluna alvo (y), por padrão 'ret'.
        # -> window : int                    Janela móvel para recalcular a regressão.
        # -> scale : bool                    Se True, normaliza as features (z-score) dentro da janela.
        # -> add_intercept : bool            Se True, adiciona intercepto automaticamente (default).
        # -> min_periods : int               Número mínimo de observações antes de iniciar o cálculo.
        # -> normalize_output : bool         Se True, normaliza a previsão final entre -1 e 1.
        # ----------

        super().__init__(asset, timeframe)
        self.features = features or ['ret']
        self.target_col = target_col
        self.window = window
        self.scale = scale
        self.add_intercept = add_intercept
        self.normalize_output = normalize_output

    def calculate(self, df: pd.DataFrame):
        df = df.copy()

        if len(df) < self.window:
            raise ValueError(f"Not enough data with window size")

        # Checar colunas
        for col in self.features + [self.target_col]:
            if col not in df.columns:
                raise ValueError(f"Coluna '{col}' não encontrada no DataFrame.")

        preds = np.full(len(df), np.nan)

        start = self.window if self.window > 1 else 0
        for i in range(start, len(df)):
            sub = df.iloc[max(0, i - self.window):i]

            X = sub[self.features].values
            y = sub[self.target_col].values

            # Normaliza dentro da janela
            if self.scale:
                X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

            # Treina regressão
            model = LinearRegression(fit_intercept=self.add_intercept)
            model.fit(X, y)

            # Prediz o próximo ponto (última linha)
            X_next = df[self.features].iloc[[i]].values
            if self.scale:
                X_next = (X_next - sub[self.features].mean().values) / (sub[self.features].std().values + 1e-8)
            preds[i] = model.predict(X_next)[0]

        # Normaliza saída
        if self.normalize_output:
            preds = (preds - np.nanmean(preds)) / (np.nanstd(preds) + 1e-8)
            preds = np.clip(preds, -1, 1)

        return preds 

# df = LinearRegressionIndicator(
#     features=['ret', 'rv', 'hawkes'],
#     target_col='ret',
#     window=63
# ).calculate(df)

# print(df[['datetime', 'linreg_pred']].tail())