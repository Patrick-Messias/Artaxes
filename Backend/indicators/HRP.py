import polars as pl
import numpy as np
from typing import Union
from scipy.cluster.hierarchy import linkage
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import sys
sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend\core')
from Indicator import Indicator # type: ignore

class HRP(Indicator):
    """
    Hierarchical Risk Parity (HRP) Allocation Indicator.
    Description:
    This indicator calculates the optimal portfolio weight for a specific asset/model 
    within a group. It uses hierarchical clustering to group assets with similar 
    risk profiles and applies recursive bisection to distribute weights based on 
    inverse variance. Unlike Mean-Variance optimization, it does not require 
    matrix inversion, making it highly robust for noisy financial data.
    """

    def __init__(self, asset=None, timeframe=None, **params):
        defaults = {
            'window': 63,
            'linkage_method': 'single', # 'single', 'complete', 'average', 'ward'
            'price_col': 'close'
        }
        defaults.update(params)
        super().__init__(asset, timeframe, **defaults)
        self.name = "hrp_weight"

    def _calculate_logic(self, data: pl.DataFrame, **kwargs) -> pl.Series:
        window = int(kwargs.get('window', 63))
        method = str(kwargs.get('linkage_method', 'single'))
        target_asset = self.asset # O asset específico deste indicador

        # 1. Garantir que temos retornos
        # O input esperado aqui é o DataFrame de preços de múltiplos ativos/modelos
        returns_df = data.select([
            pl.col(c).pct_change().fill_null(0.0) for c in data.columns 
            if c not in ['ts', 'datetime']
        ])
        
        asset_names = returns_df.columns
        if target_asset not in asset_names:
            # Fallback caso o nome do asset não esteja nas colunas
            return pl.Series([0.0] * len(data)).alias(self.name)

        target_idx = asset_names.index(target_asset)
        numpy_returns = returns_df.to_numpy()
        num_rows = len(numpy_returns)
        weights_history = np.zeros(num_rows)

        # 2. Funções auxiliares do HRP (Padrão de Prado)
        def get_ivp(cov):
            # Variance Inversion
            ivp = 1.0 / np.diag(cov)
            ivp /= ivp.sum()
            return ivp

        def get_cluster_var(cov, c_items):
            # Variance of a cluster
            cov_c = cov[np.ix_(c_items, c_items)]
            w_ = get_ivp(cov_c)
            c_var = np.dot(np.dot(w_, cov_c), w_)
            return c_var

        def get_quasi_diag(link):
            # Sort clusters
            link = link.astype(int)
            sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
            num_items = link[-1, 3]
            while sort_ix.max() >= num_items:
                sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
                df0 = sort_ix[sort_ix >= num_items]
                i = df0.index
                j = df0.values - num_items
                sort_ix[i] = link[j, 0]
                df1 = pd.Series(link[j, 1], index=i + 1)
                sort_ix = pd.concat([sort_ix, df1]).sort_index()
            return sort_ix.tolist()

        import pandas as pd # Usado internamente apenas para o sort_ix do HRP

        # 3. Loop deslizante (como o HRP é matricial, rodamos via NumPy para performance)
        for t in range(window, num_rows):
            window_slice = numpy_returns[t-window:t]
            
            # Covariância e Correlação
            cov = np.cov(window_slice, rowvar=False)
            corr, _ = spearmanr(window_slice)
            
            # Distância baseada em correlação
            dist = np.sqrt((1.0 - corr) / 2.0)
            np.fill_diagonal(dist, 0)
            
            # Linkage e Quasi-Diagonalização
            # dist precisa ser condensada para o linkage
            from scipy.spatial.distance import squareform
            link = linkage(squareform(dist), method=method)
            sort_ix = get_quasi_diag(link)
            
            # Bisseção Recursiva
            weights = np.ones(len(asset_names))
            items = [sort_ix]
            
            while len(items) > 0:
                items = [items[i][j:k] for i in range(len(items)) 
                         for j, k in ((0, len(items[i]) // 2), (len(items[i]) // 2, len(items[i])))]
                
                for i in range(0, len(items), 2):
                    c_left = items[i]
                    c_right = items[i+1]
                    
                    if len(c_left) > 0 and len(c_right) > 0:
                        var_l = get_cluster_var(cov, c_left)
                        var_r = get_cluster_var(cov, c_right)
                        alpha = 1 - var_l / (var_l + var_r)
                        
                        weights[c_left] *= alpha
                        weights[c_right] *= (1 - alpha)
                
                # Remove listas unitárias (folhas da árvore)
                items = [i for i in items if len(i) > 1]
            
            weights_history[t] = weights[target_idx]

        return pl.Series(weights_history).fill_null(0.0).alias(self.name)

'''
# ==========================================
# MAIN TEMPORÁRIA PARA TESTE
# ==========================================
if __name__ == "__main__":
    # 1. Gerar dados sintéticos (3 modelos correlacionados)
    np.random.seed(42)
    n_obs = 500
    
    # Modelo 1: Base
    m1 = np.random.normal(0.0001, 0.01, n_obs)
    # Modelo 2: Correlacionado com M1 (60%)
    m2 = 0.6 * m1 + 0.4 * np.random.normal(0.0001, 0.01, n_obs)
    # Modelo 3: Baixa correlação e mais volátil
    m3 = np.random.normal(0.0001, 0.03, n_obs)
    
    # Converter retornos em "Preços" para o indicador
    df_prices = pl.DataFrame({
        "Model_A": np.exp(np.cumsum(m1)),
        "Model_B": np.exp(np.cumsum(m2)),
        "Model_C": np.exp(np.cumsum(m3))
    })

    print("--- Calculando HRP Weights para cada modelo ---")
    
    # Calcular pesos para o Model A e Model C (os mais diferentes)
    hrp_a = HRP(asset="Model_A", window=21)
    hrp_b = HRP(asset="Model_B", window=21)
    hrp_c = HRP(asset="Model_C", window=21)

    res_a = hrp_a._calculate_logic(df_prices)
    res_b = hrp_b._calculate_logic(df_prices)
    res_c = hrp_c._calculate_logic(df_prices)

    print(res_a)

    # Plotar
    plt.figure(figsize=(12, 6))
    plt.plot(res_a.to_numpy(), label="HRP Weight: Model A (Estável/Corr)", color='blue')
    plt.plot(res_b.to_numpy(), label="HRP Weight: Model B (Corr)", color='green')
    plt.plot(res_c.to_numpy(), label="HRP Weight: Model C (Volátil/Indep)", color='red')
    plt.title("HRP Weight Allocation Over Time")
    plt.axhline(0.33, color='gray', linestyle='--', label="Equal Weight (1/3)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

'''

