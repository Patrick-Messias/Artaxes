import polars as pl, numpy as np, seaborn as sns, matplotlib.pyplot as plt, datetime
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from typing import Union, Optional, Dict, List

def Sharpe(): 
    return None

def ExpectedValue():
    return None

def TimeValitation():
    # ?
    return None

def Score(data: Union[pl.DataFrame, Dict, List], weight: Optional[list]=None, mc_params: Dict={'runs': 1000, 'shuffle': True, 'select_col': 'wfm_matrix_data', 'percentile': 95.0}):
    if weight is None: weight = [0.25, 0.25, 0.25, 0.25]

    # scoreA = (wfe_sharpe * weight[0]) * (expected_value * weight[1]) * (1/monte_carlo_drawdown * weight[2]) * (time_validation * weight[3])
    # scoreB = (wfe_sharpe * weight[0]) / (time_validation * weight[3]) + (expected_value * weight[1]) / (1/monte_carlo_drawdown * weight[2])

    return None # (scoreA + scoreB) / 2

def MonteCarlo(data: Union[pl.DataFrame, Dict, List], runs: int=1000, shuffle: bool = True, select_col: str='wfm_matrix_data') -> np.ndarray:
    # data: DataFrame Polars (WFM Matrix), Dict or list of returnsa
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


def mc_drawdown(mc_runs: np.ndarray, percentile: float=95.0, only_max_per_curve: bool=True):
    equities = np.cumsum(mc_runs, axis=1)
    peaks = np.maximum.accumulate(equities, axis=1)
    drawdowns = equities - peaks
    
    # Pior drawdown de cada simulação
    max_drawdowns = np.min(drawdowns, axis=1)
    
    # Todos drawdowns (filtrando só negativos)
    intra_drawdowns = drawdowns[drawdowns < 0]
    if intra_drawdowns.size == 0:
        intra_drawdowns = np.array([0.0])

    # Escolha do dataset
    data = max_drawdowns if only_max_per_curve else intra_drawdowns

    # VaR / CVaR
    var_level = 100 - percentile
    avg = np.mean(data)
    var = np.percentile(data, var_level)
    cvar = data[data <= var].mean()

    return {
        "mean": avg,
        "var": var,
        "cvar": cvar
    }

def mc_stagnation(mc_runs: np.ndarray, percentile: float=95.0, only_max_per_curve: bool=True):
    equities = np.cumsum(mc_runs, axis=1)
    peaks = np.maximum.accumulate(equities, axis=1)
    is_at_peak = (equities == peaks)

    num_days = equities.shape[1]

    max_stags = []
    all_stags = []

    for run_peaks in is_at_peak:
        peak_days = np.where(run_peaks)[0]
        durations = np.diff(np.concatenate(([0], peak_days, [num_days-1])))
        durations = durations[durations > 0]

        if len(durations) > 0:
            max_stags.append(np.max(durations))
            all_stags.extend(durations)
        else:
            max_stags.append(0)

    max_stags = np.array(max_stags)
    all_stags = np.array(all_stags) if len(all_stags) > 0 else np.array([0.0])

    data = max_stags if only_max_per_curve else all_stags

    var_level = percentile
    avg = np.mean(data)
    perc = np.percentile(data, var_level)

    return {
        "mean": avg,
        "perc": perc
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


# Permutation Test - How model compares to random data sample?


# In Sample - Out Sample Test
def ISOS_TEST(data: pl.DataFrame, os_start_datetime=None, os_end_datetime=None, metrics: list=['pnl', 'dd', 'pnl_dd'], mc_runs: int=1000, mc_shuffle: bool=True, mc_col: str='wfm_matrix_data', datetime_format: str="%Y-%m-%d %H:%M:%S"):

    if os_start_datetime is None: 
        print("< [Error] No start date for OS.")
        return []
    
    last_dt = data["datetime"].last()
    if os_start_datetime >= last_dt or os_end_datetime > last_dt:
        print("< [Error] Start or end datetime higher then last datetime on data")
        return []
    
    # If str, converts to datetime
    dt_start = datetime.strptime(os_start_datetime, datetime_format) if isinstance(os_start_datetime, str) else os_start_datetime
    dt_end = datetime.strptime(os_end_datetime, datetime_format) if isinstance(os_end_datetime, str) else os_end_datetime
    if dt_end < dt_start:
        print("< [Error] End datetime is before start datetime")
        return []
    
    # Selects OS Data
    os_df = data.filter(pl.col("datetime").is_between(os_start_datetime, os_end_datetime))
    is_df = data.filter(pl.col("datetime") < os_end_datetime)

    # Runs MC on len of os_data from is_data
    os_len = int(os_df.height) # To select sample from is_mc_data and compare
    is_mc_data = MonteCarlo(is_df['pnl'], mc_runs, mc_shuffle, mc_col)

    # Analyzes os_df performance with random os_len sized samples from is_mc_data
    os_pnl = np.sum(os_df['pnl'])
    for rn in mc_runs:
        pass # GET mc_runs number of random samples of os_len size
    
    for met in metrics:
        pass # COMPARES sample to data

    return None



# Entry Cluster Analysis - Are entries/signals grouped or distributed across datetime? 








































