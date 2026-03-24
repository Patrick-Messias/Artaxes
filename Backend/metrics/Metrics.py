import polars as pl, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from typing import Union, Optional, Dict, List

def MonteCarlo(data: Union[pl.DataFrame, Dict, List], runs: int=1000, shuffle: bool = True, select_col: str='wfm_matrix_data') -> np.ndarray:
    # data: DataFrame Polars (WFM Matrix), Dict or list of returns
    # runs: Quantity of simulations to be generated
    # shuffle: True for Permutation (without reposition), False for Bootstrap (with reposition)
    # select_col: Name of the profit/loss column if DataFrame
    # Returns np.ndarray: Matrix of (runs * n_days) with data

    # Extracts and Normalizes numpy data
    if isinstance(data, pl.DataFrame): # If is complete wfm_matrix takes pnl col
        val_array = data[select_col].to_numpy()
    elif isinstance(data, dict): # If is results dict (ex: {ps_id: [returns]}), decides if wants average ou specific
        val_array = np.array(list(data.values()) if not isinstance(list(data.values())[0], list) 
                                    else [item for sublist in data.values() for item in sublist])
    else:
        val_array = np.array(data)

    n_samples = len(val_array)

    # Index generation matrix
    if shuffle: # Permutation
        inds = np.array([np.random.permutation(n_samples) for _ in range(runs)])
    else: # Bootstrap
        inds = np.random.choice(n_samples, size=(runs, n_samples), replace=True)

    # Returns mapping and equity calculation
    mc_runs = val_array[inds]

    return mc_runs

def get_mc_metrics(mc_runs: np.ndarray, percentile: float=95.0):
    # mc_runs: (runs, days) -> Brute returns from MonteCarlo

    # Generates equity curves to simulate drawdown and total retuns
    equities = np.cumsum(mc_runs, axis=1)

    # Max Drawdown of every simulation
    peaks = np.maximum.accumulate(equities, axis=1)
    drawdowns = equities - peaks # Negative values
    max_drawdowns = np.min(drawdowns, axis=1) # Worst drawdown of every run

    # Final return of every simulation
    final_returns = equities[:,-1]

    # Drawdown statistics, orders drawdowns (best-worst)
    sorted_dd = np.sort(max_drawdowns)
    idx = int((1-percentile/100) * len(sorted_dd))

    var_dd = sorted_dd[idx]                 # VaR
    cvar_dd = np.mean(sorted_dd[:idx+1])    # CVaR (Avg of worsts)
    avg_dd = np.mean(max_drawdowns)         # Average Drawdown

    # Returns Statistics
    avg_win = np.percentile(final_returns, 50) # Median (percentile 50)

    return {
        "avg_drawdown": avg_dd,
        "drawdown_var_95": var_dd,
        "drawdown_cvar_95": cvar_dd,
        "avg_win_median": avg_win,
        "worst_run": np.min(final_returns),
        "best_run": np.max(final_returns)
    }


# WIP Period Returns Correlation Clustering with Monte Carlo - Model-Strat-Asset-Parset/WF
def CorrelationClusteringMC(
    basket_results: Dict[str, List[float]], 
    resolution_days: int = 63, # ~3 meses
    mc_params: Optional[Dict] = None, # [runs, shuffle, pnl_col, percentile] -> [10000, True, 'wfm_matrix_data', 95.0]
    metric_type: str = "pnl_sum"
):
    """
    Gera um ClusterMap (Dendrograma + Heatmap) comparando os itens da cesta.
    basket_results: { 'Nome_Modelo_1': [retornos_diarios], 'Nome_Modelo_2': [...] }
    """
    
    analysis_matrix = {}
    
    for name, returns in basket_results.items():
        # 1. Preparação dos dados
        data = np.array(returns)
        n_days = len(data)
        
        # 2. Janelamento (Rolling Windows)
        windows = []
        for i in range(0, n_days - resolution_days + 1, resolution_days):
            window_slice = data[i : i + resolution_days]
            
            # 3. Aplicação opcional de Monte Carlo por Janela
            if mc_params:
                # Geramos os caminhos MC apenas para esta janela
                # mc_params: {'runs': 1000, 'shuffle': True, 'percentile': 50}
                mc_runs = MonteCarlo(
                    data=window_slice, 
                    runs=mc_params['runs'], 
                    shuffle=mc_params['shuffle']
                )
                
                # Calculamos a métrica em cada simulação (ex: PnL acumulado da janela)
                if metric_type == "pnl_sum":
                    mc_metrics = np.sum(mc_runs, axis=1)
                elif metric_type == "max_dd":
                    # Cálculo rápido de DD para MC
                    equities = np.cumsum(mc_runs, axis=1)
                    picos = np.maximum.accumulate(equities, axis=1)
                    mc_metrics = np.min(equities - picos, axis=1)
                
                # Pegamos o valor representativo (percentile 50 = mediana)
                val = np.percentile(mc_metrics, mc_params.get('percentile', 50))
            else:
                # Sem MC: Calculamos a métrica simples da janela
                val = np.sum(window_slice) if metric_type == "pnl_sum" else np.min(np.cumsum(window_slice) - np.maximum.accumulate(np.cumsum(window_slice)))
            
            windows.append(val)
        
        analysis_matrix[name] = windows

    # 4. Criação do DataFrame para Clustering
    df_cluster = pl.DataFrame(analysis_matrix).to_pandas().T
    # df_cluster agora tem: Linhas = Modelos/Estratégias, Colunas = Janelas de Tempo
    
    # 5. Cálculo da Correlação e Clustering Hierárquico
    # Usamos correlação de correlação para a distância (1 - corr)
    corr_matrix = df_cluster.T.corr()
    dist_matrix = 1 - corr_matrix
    
    # Linkage (Agrupamento) usando o método de Ward (minimiza variância interna)
    linkage_matrix = hierarchy.linkage(squareform(dist_matrix), method='ward')

    # 6. Visualização (ClusterMap)
    plt.style.use('dark_background')
    g = sns.clustermap(
        corr_matrix,
        row_linkage=linkage_matrix,
        col_linkage=linkage_matrix,
        cmap="RdYlGn", # Verde para correlacionado, Vermelho para descorrelacionado
        annot=True,
        fmt=".2f",
        figsize=(12, 10),
        cbar_kws={'label': 'Correlação de Pearson'}
    )
    
    g.fig.suptitle(f"Hierarchical Clustering | Metric: {metric_type} | Windows: {resolution_days}d", 
                   color='white', fontsize=14, y=1.02)
    
    plt.show()
    
    return linkage_matrix, corr_matrix



# Permutation Test














































