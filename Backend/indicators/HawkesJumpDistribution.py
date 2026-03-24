import polars as pl
import numpy as np
from scipy.optimize import minimize

class HawkesJumpIndicator:
    def __init__(self, window_size: int = 252):
        self.window_size = window_size

    def _hawkes_log_likelihood(self, params, t):
        """
        Log-Likelihood simplificada para um processo de Hawkes (Kernel Exponencial)
        mu: taxa base, alpha: intensidade do choque, beta: taxa de decaimento
        """
        mu, alpha, beta = params
        if mu <= 0 or alpha < 0 or beta <= alpha: # Estabilidade: beta > alpha
            return 1e10
        
        n = len(t)
        dt = np.diff(t)
        
        # Intensidade recursiva
        r = np.zeros(n)
        for i in range(1, n):
            r[i] = (r[i-1] + alpha) * np.exp(-beta * dt[i-1])
            
        lambda_t = mu + r
        log_lik = - (mu * t[-1]) - np.sum(alpha/beta * (1 - np.exp(-beta * (t[-1] - t)))) + np.sum(np.log(lambda_t))
        return -log_lik

    def calculate_risk_score(self, returns_slice: np.ndarray):
        """
        Analisa a distribuição: 
        1. Identifica 'Jumps' (retornos > 2.5 std dev)
        2. Ajusta Hawkes nos timestamps dos Jumps
        """
        # 1. Identificação de Jumps (Eventos de cauda)
        std = np.std(returns_slice)
        threshold = 2.5 * std
        jump_indices = np.where(np.abs(returns_slice) > threshold)[0]
        
        if len(jump_indices) < 3:
            return 0.0 # Sem eventos suficientes para modelar excitação
        
        # Timestamps fictícios para os eventos dentro da janela
        t = jump_indices.astype(float)
        
        # 2. Fit Hawkes Process
        initial_guess = [0.1, 0.5, 1.0]
        res = minimize(self._hawkes_log_likelihood, initial_guess, args=(t,), 
                       bounds=[(1e-4, None), (0, None), (1e-4, None)], method='L-BFGS-B')
        
        mu, alpha, beta = res.x
        
        # O 'Risk Score' é a Intensidade Atual (Excitação + Base)
        # Se alpha/beta é alto, o risco de efeito cascata é maior
        excitation_factor = alpha / (beta - alpha + 1e-9)
        return excitation_factor

    def apply(self, df: pl.DataFrame, col_name: str = "close"):
        """
        Aplica o indicador em uma rolling window
        """
        # Calcula log-retornos
        df = df.with_columns(
            returns = pl.col(col_name).log().diff()
        ).drop_nulls()

        # Função auxiliar para o polars
        def rolling_logic(s: pl.Series) -> float:
            return self.calculate_risk_score(s.to_numpy())

        return df.with_columns(
            hawkes_jump_risk = pl.col("returns").rolling_map(
                rolling_logic, 
                window_size=self.window_size
            )
        )

# --- Exemplo de Uso ---
if __name__ == "__main__":
    # Simulando dados com clusters de volatilidade (Jumps)
    np.random.seed(42)
    data = np.random.normal(0, 0.01, 1000)
    data[400:420] += np.random.normal(0, 0.05, 20) # Cluster de choque 1
    data[800:810] += np.random.normal(0, 0.08, 10) # Cluster de choque 2
    
    df = pl.DataFrame({"close": np.exp(np.cumsum(data))})
    
    indicator = HawkesJumpIndicator(window_size=100)
    df_result = indicator.apply(df)

    clipped_df = df_result.with_columns(
        pl.col("hawkes_jump_risk").clip(upper_bound=1)
    )
    
    print(clipped_df.tail(20))
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(14, 6))
    plt.plot(clipped_df)
    plt.tight_layout()
    plt.show()








