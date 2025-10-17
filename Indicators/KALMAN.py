import pandas as pd, numpy as np
from Indicator import Indicator

"""
Kalman Filter: is a recursive algorithm used to estimate the state of a dynamic system from a series of noisy measurements.
It is widely used in time series analysis and signal processing to smooth data, predict future values, and filter out noise.

Formula (Assumes linear system):
xk​ = Axk−1 + Buk​ + wk​
zk​ = Hxk​ + vk​
where:
xk​: True state at time k (real price)
zk​: Observed measurement (observed price)
A: State transition model (how the state evolves)
B: Control input model (how control inputs affect the state)
H: Observation model (how the state maps to observations)
wk, vk: Process and measurement noise (~N(0, Q) and ~N(0, R))
"""

import pandas as pd
import numpy as np
from Indicator import Indicator

class KALMAN(Indicator): 
    def __init__(self, asset, timeframe: str, R: list = [0.001, 0.01+0.01, 0.001], Q: list = [0.001, 0.001+0.001, 0.001], price_col: str = 'close'):
        super().__init__(asset, timeframe)
        self.R = R if R is not None else [0.01]  # Lista de measurement noise variances
        self.Q = Q if Q is not None else [0.001]  # Lista de process noise variances
        self.price_col = price_col

    def calculate(self, df: pd.DataFrame, R: float, Q: float) -> pd.Series:        
        z = df[self.price_col].values
        n = len(z)
        x_est = np.zeros(n)
        P = np.zeros(n)
    
        # Init
        x_est[0] = z[0]
        P[0] = 1.0

        # Filter iterations
        for k in range(1, n):
            # Prediction
            x_pred = x_est[k-1]
            P_pred = P[k-1] + Q

            # Update
            K = P_pred / (P_pred + R)
            x_est[k] = x_pred + K * (z[k] - x_pred)
            P[k] = (1 - K) * P_pred

        return pd.Series(x_est, index=df.index, name=f'kalman_R{R}_Q{Q}')

    def calculate_all_sets(self, df: pd.DataFrame) -> dict:
        """
        Retorna um dicionário com todas as combinações de parâmetros R e Q
        Similar ao comportamento do MA e HURST
        """
        results = {}
        
        for r_val in self.R:
            for q_val in self.Q:
                param_set = f"R{r_val}_Q{q_val}"
                results[param_set] = self.calculate(df, R=r_val, Q=q_val)
        
        return results



"""
class KALMAN(Indicator): 
    def __init__(self, asset, timeframe: str, R: float = 0.01, Q: float = 0.001, price_col: str = 'close'):
        super().__init__(asset, timeframe)
        self.R = R  # Measurement noise variance
        self.Q = Q  # Process noise variance
        self.price_col = price_col

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:        
        z = df[self.price_col].values
        n = len(z)
        x_est = np.zeros(n)
        P = np.zeros(n)
    
        # Init
        x_est[0] = z[0]
        P[0] = 1.0

        # Filter iterations
        for k in range(1, n):
            # Prediction
            x_pred = x_est[k-1]
            P_pred = P[k-1] + self.Q

            # Update
            K = P_pred / (P_pred + self.R)
            x_est[k] = x_pred + K * (z[k] - x_pred)
            P[k] = (1 - K) * P_pred

        return pd.Series(x_est, index=df.index, name=f'kalman_{self.price_col}')

"""




