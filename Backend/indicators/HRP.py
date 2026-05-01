import polars as pl, numpy as np, pandas as pd, sys
from typing import Union
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

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
            'price_col': 'close',
            'is_returns': True
        }
        defaults.update(params)
        super().__init__(asset, timeframe, **defaults)
        self.name = "hrp_weight"

    def _calculate_logic(self, data: pl.DataFrame, **kwargs) -> Union[pl.Series, dict]:
        # 1. Sanitização de Entrada (Resolve o erro 'list.contains')
        # Forçamos todas as colunas numéricas a serem Float64 e removemos lixo
        df = data.select([
            pl.col(c).cast(pl.Float64, strict=False) 
            for c in data.columns if c not in ['ts', 'datetime']
        ]).fill_null(0.0).fill_nan(0.0)

        window = int(kwargs.get('window', self.params.get('window', 63)))
        method = str(kwargs.get('linkage_method', self.params.get('linkage_method', 'single')))
        is_returns = kwargs.get('is_returns', self.params.get('is_returns', True))
        target_asset = self.asset 

        # 2. Garantir que temos retornos (Apenas se a entrada for Preço)
        if not is_returns:
            returns_df = df.select([
                pl.col(c).pct_change().fill_null(0.0) for c in df.columns
            ])
        else:
            returns_df = df

        asset_names = returns_df.columns
        numpy_returns = returns_df.to_numpy()
        num_rows = len(numpy_returns)
        
        # Se o dataframe for menor que a janela, ajustamos a janela
        effective_window = min(window, num_rows)

        # --- FUNÇÕES CORE HRP (Mantidas conforme Prado) ---
        def get_ivp(cov):
            variance = np.diag(cov)
            # Evita divisão por zero em ativos sem volatilidade
            variance = np.where(variance <= 0, 1e-9, variance)
            ivp = 1.0 / variance
            ivp /= ivp.sum()
            return ivp

        def get_cluster_var(cov, c_items):
            cov_c = cov[np.ix_(c_items, c_items)]
            w_ = get_ivp(cov_c)
            c_var = np.dot(np.dot(w_, cov_c), w_)
            return c_var

        def get_quasi_diag(link):
            link = link.astype(int)
            sort_ix = [link[-1, 0], link[-1, 1]]
            num_items = link[-1, 3]
            while max(sort_ix) >= num_items:
                new_sort_ix = []
                for x in sort_ix:
                    if x >= num_items:
                        idx = x - num_items
                        new_sort_ix.extend([link[idx, 0], link[idx, 1]])
                    else:
                        new_sort_ix.append(x)
                sort_ix = new_sort_ix
            return sort_ix

        # 3. Execução do HRP
        # Se estamos no modo Manager (Rebalance), calculamos apenas para a última janela
        # Se estamos no modo Indicador (Backtest), calculamos o histórico
        
        def calculate_at_point(matrix_slice):
            # Garante que matrix_slice seja 2D (Linhas, Colunas)
            if matrix_slice.ndim == 1:
                matrix_slice = matrix_slice.reshape(-1, 1)
            
            # Se tivermos apenas 1 linha de dados, não há como calcular correlação/covariância
            if matrix_slice.shape[0] < 2:
                return np.ones(len(asset_names)) / len(asset_names)

            # 1. Matriz de Correlação Spearman
            # Forçamos o cálculo entre colunas (axis=0)
            corr_res = spearmanr(matrix_slice, axis=0)
            
            # O spearmanr pode retornar um escalar se houver apenas 2 colunas em versões antigas
            # ou um objeto com o atributo .statistic
            if hasattr(corr_res, 'statistic'):
                corr = corr_res.statistic
            else:
                corr = corr_res[0] if isinstance(corr_res, tuple) else corr_res

            # Se a correlação vier como escalar (comum em matrizes 2x2), transformamos em matriz
            if np.isscalar(corr):
                corr = np.array([[1.0, corr], [corr, 1.0]])

            if np.isnan(corr).all(): 
                return np.ones(len(asset_names)) / len(asset_names)
            
            # 2. Matriz de Distância
            dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0, 1))
            np.fill_diagonal(dist, 0)
            
            # 3. Covariância
            cov = np.cov(matrix_slice, rowvar=False)
            # Garante que cov seja matriz mesmo com 2 ativos
            if np.isscalar(cov):
                cov = np.array([[cov]])

            # 4. Clustering e Pesos
            try:
                link = linkage(squareform(dist), method=method)
                sort_ix = get_quasi_diag(link)
                
                weights = np.ones(len(asset_names))
                items = [sort_ix]
                
                while len(items) > 0:
                    items = [items[i][j:k] for i in range(len(items)) 
                             for j, k in ((0, len(items[i]) // 2), (len(items[i]) // 2, len(items[i])))]
                    for i in range(0, len(items), 2):
                        c_l, c_r = items[i], items[i+1]
                        if len(c_l) > 0 and len(c_r) > 0:
                            var_l = get_cluster_var(cov, c_l)
                            var_r = get_cluster_var(cov, c_r)
                            alpha = 1 - var_l / (var_l + var_r)
                            weights[c_l] *= alpha
                            weights[c_r] *= (1 - alpha)
                    items = [i for i in items if len(i) > 1]
                return weights
            except Exception:
                # Se o clustering falhar (ex: distância zero), retorna Equal Weight
                return np.ones(len(asset_names)) / len(asset_names)

        # Decisão de Retorno:
        if target_asset is None:
            # MODO MANAGER: Retorna dicionário de pesos do último ponto disponível
            final_weights = calculate_at_point(numpy_returns[-effective_window:])
            return dict(zip(asset_names, final_weights))
        else:
            # MODO INDICADOR: Retorna pl.Series histórica para o target_asset
            weights_history = np.zeros(num_rows)
            target_idx = asset_names.index(target_asset) if target_asset in asset_names else 0
            
            # Rodar apenas do window em diante para performance
            for t in range(effective_window, num_rows + 1):
                window_slice = numpy_returns[max(0, t-effective_window):t]
                if window_slice.shape[0] < 2: continue
                w_t = calculate_at_point(window_slice)
                if t < num_rows:
                    weights_history[t] = w_t[target_idx]
                else:
                    # Caso seja o ponto exato final
                    weights_history[-1] = w_t[target_idx]
            
            return pl.Series(weights_history).fill_null(0.0).alias(self.name)


