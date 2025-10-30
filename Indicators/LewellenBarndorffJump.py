import pandas as pd
import numpy as np
from Indicator import Indicator

class LewellenBarndorffJump(Indicator):
  
    # Jump Indicator baseado nos testes de Lewellen/Barndorff-Jump.
    # Detecta movimentos abruptos (jumps) no preço, removendo a volatilidade contínua.
  
    def __init__(self, asset=None, timeframe: str=None, window: int=21, price_col: str='close', normalized: float=None): #1e-8
        super().__init__(asset, timeframe)
        self.price_col = price_col
        self.window = window  # janela para estimativa de volatilidade contínua
        self.normalized = normalized

    def calculate(self, df: pd.DataFrame):
        df_ = df.copy()
        if self.price_col not in df_.columns:
            raise ValueError(f"Coluna '{self.price_col}' não encontrada no DataFrame.")

        # 1️⃣ Log-retornos
        df_['log_ret'] = np.log(df_[self.price_col] / df_[self.price_col].shift(1)).fillna(0)

        # 2️⃣ Estimar volatilidade contínua usando Parkinson ou Realized Vol
        if 'rv' not in df_.columns: df_['rv'] = np.log(df_['high'] / df_['low']).replace([np.inf, -np.inf], np.nan).fillna(0)
        df_['rv_roll_std'] = df_['rv'].rolling(self.window).std().fillna(0)

        # 3️⃣ Jump: diferença entre log-retorno absoluto e volatilidade contínua
        df_['jump'] = np.maximum(0, np.abs(df_['log_ret']) - df_['rv_roll_std'])

        # 4️⃣ Normalizar (opcional, para escala compatível com HMM)
        if self.normalized is not None:
            df_['jump'] = df_['jump'] / (df_['rv_roll_std'] + self.normalized) 
        return df_['jump']
