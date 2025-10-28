import sys
sys.path.append(f'C:\\Users\\Patrick\\Desktop\\ART_Backtesting_Platform\\Backend')

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from typing import Literal
from tqdm import tqdm
from Indicator import Indicator
import pandas as pd, numpy as np

class GBHAR(Indicator):
    def __init__(self, 
                 asset, 
                 timeframe: str, 
                 lookback_window: float = 252,
                 predict_window: int = 21,
                 predict: Literal['vol_avg', 'direction'] = 'vol_avg',  # 'vol_avg' ou 'direction'
                 predict_n: int = 1, # n=1 predict next period, n=5 predicts the 5th period ahead
                 exogenous_vars: dict = None,
                 return_columns: list = ['prediction'],
                 rolling_mode: Literal['expanding', 'rolling'] = 'rolling', # 'expanding' ou 'rolling'
                 price_col_func: callable = None,  # aceita função ex: lambda df: (df['high'] - df['low']).ffill()
                 gradient_boosting_regressor_params: dict = None):
        
        super().__init__(asset, timeframe)
        self.lookback_window = lookback_window
        self.predict_window = predict_window
        self.predict = predict
        self.predict_n = predict_n
        self.exogenous_vars = exogenous_vars
        self.return_columns = return_columns
        self.rolling_mode = rolling_mode
        self.price_col_func = price_col_func or (lambda df: np.log(df['high'] / df['low']).replace([np.inf, -np.inf], np.nan).ffill())
        self.gradient_boosting_regressor_params = gradient_boosting_regressor_params or {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 3,
            'subsample': 0.9,
            'random_state': 42
        }

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Volatilidade ou direção
        df['rv'] = self.price_col_func(df)
        if self.predict == 'direction':
            df['target'] = (df['close'] - df['open'] > 0).astype(int)
        else:
            df['target'] = df['rv']

        # Features HAR
        df['rv_lag1'] = df['rv'].shift(1)
        df['rv_week'] = df['rv'].rolling(5).mean().shift(1)
        df['rv_month'] = df['rv'].rolling(21).mean().shift(1)
        df.dropna(subset=['rv', 'rv_lag1', 'rv_week', 'rv_month'], inplace=True)

        X = df[['rv_lag1', 'rv_week', 'rv_month']].astype(np.float32)

        # Adiciona exogenous_vars se fornecidas
        if self.exogenous_vars:
            for key, series in self.exogenous_vars.items():
                X[key] = series.loc[df.index]

        y = df['target'].astype(np.float32)

        # Modelo
        gb = GradientBoostingRegressor(**self.gradient_boosting_regressor_params)
        model_name = "GradientBoostingRegressor (CPU)"

        # Ajusta lookback_window
        lookback = int(self.lookback_window) if self.lookback_window >= 1 else int(len(df) * self.lookback_window)
        lookback = max(10, lookback)

        # Rolling forecast + rolling accuracy
        predictions = [np.nan] * len(df)
        rolling_accuracy = [np.nan] * len(df)

        for i in tqdm(range(lookback, len(df) - self.predict_n),
                      desc=f'Rolling forecast ({model_name})'):
            start_idx = 0 if self.rolling_mode == 'expanding' else i - lookback
            X_train = X.iloc[start_idx:i]
            y_train = y.iloc[start_idx:i]
            X_pred = X.iloc[i + self.predict_n - 1:i + self.predict_n]

            if len(X_train) < 10 or X_pred.empty:
                continue

            gb.fit(X_train, y_train)
            pred = gb.predict(X_pred)[0]
            predictions[i + self.predict_n - 1] = pred

            # Atualiza rolling accuracy até o ponto i
            if self.predict == 'direction':
                y_true_cum = (y.iloc[start_idx:i+1] > 0).astype(int)
                y_pred_cum = (predictions[start_idx:i+1] > 0).astype(int)
                y_pred_cum = pd.Series(y_pred_cum).fillna(0)
                rolling_accuracy[i + self.predict_n - 1] = accuracy_score(y_true_cum, y_pred_cum)
            else:
                # Binário vol > média
                df['avg_vol'] = df['rv'].rolling(self.predict_window).mean()
                y_true_cum = (df['rv'].iloc[start_idx:i+1] > df['avg_vol'].iloc[start_idx:i+1]).astype(int)
                y_pred_cum = (np.array(predictions[start_idx:i+1]) > df['avg_vol'].iloc[start_idx:i+1]).astype(int)
                y_pred_cum = pd.Series(y_pred_cum).fillna(0)
                rolling_accuracy[i + self.predict_n - 1] = accuracy_score(y_true_cum, y_pred_cum)

        df['prediction'] = predictions
        df['rolling_accuracy'] = rolling_accuracy

        # Métricas finais
        valid_idx = ~pd.isna(df['prediction'])
        if valid_idx.sum() > 10:
            print(f"\n=== GB-HAR Rolling Performance ({model_name}) ===")
            if self.predict == 'direction':
                acc = accuracy_score((y[valid_idx] > 0).astype(int), (df.loc[valid_idx, 'prediction'] > 0).astype(int))
                r2 = r2_score(y[valid_idx], df.loc[valid_idx, 'prediction'])
                print(f"Accuracy: {acc:.4f}, R²: {r2:.4f}")
            else:
                acc = accuracy_score((df.loc[valid_idx, 'rv'] > df.loc[valid_idx, 'avg_vol']).astype(int),
                                     (df.loc[valid_idx, 'prediction'] > df.loc[valid_idx, 'avg_vol']).astype(int))
                r2 = r2_score(y[valid_idx], df.loc[valid_idx, 'prediction'])
                print(f"Accuracy (vol > avg_vol): {acc:.4f}, R²: {r2:.4f}")

        # ------------------------
        # Colunas finais
        # ------------------------
        if self.predict == 'vol_avg':
            df['avg_vol'] = df['rv'].rolling(self.predict_window).mean()
            df['higher'] = (df['prediction'] > df['avg_vol']).astype(int)
        elif self.predict == 'direction':
            df['higher'] = (df['prediction'] > 0).astype(int)
        else:
            raise ValueError("Predict must be 'vol_avg' or 'direction'")

        df['prediction_perc'] = (df['prediction'] - df['rv']) / df['rv']
        df['mse'] = (df['prediction'] - df['rv']) ** 2

        if self.predict == 'vol_avg': return df[self.return_columns + ['avg_vol', 'rv', 'higher', 'prediction_perc', 'mse', 'rolling_accuracy']].fillna(0)
        elif self.predict == 'direction': return df[self.return_columns + ['rv', 'higher', 'prediction_perc', 'mse', 'rolling_accuracy']].fillna(0)


# ------------------------
# Exemplo de uso
# ------------------------
df = pd.read_excel(f'C:\\Users\\Patrick\\Desktop\\Artaxes Portfolio\\MAIN\\MT5_Dados\\WIN$_D1.xlsx')
model = GBHAR(asset='WIN$', timeframe='D1', lookback_window=21, predict_window=21, rolling_mode='rolling', predict='vol_avg')
out = model.calculate(df)
df = df.join(out).fillna(0)
print(df)
