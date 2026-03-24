import polars as pl
import numpy as np
from scipy.spatial.distance import cdist
from Indicator import Indicator

class LyapunovExp(Indicator):
    """
    Indicador do Maior Expoente de Lyapunov (MLE) para séries temporais.

    Conceito:
        Os expoentes de Lyapunov medem a taxa média de separação (ou convergência) de
        trajetórias infinitesimalmente próximas no espaço de fases. Em sistemas caóticos,
        pequenas diferenças iniciais crescem exponencialmente, resultando em um expoente
        positivo. O maior expoente (MLE) domina a dinâmica e indica a sensibilidade às
        condições iniciais.

    Aplicação a séries temporais:
        Dada uma série de retornos {r_t}, reconstruímos o espaço de estados usando a técnica
        de atraso (embedding de Takens):
            Y_i = [r_i, r_{i+τ}, r_{i+2τ}, ..., r_{i+(m-1)τ}]
        onde m é a dimensão de embedding e τ é o atraso temporal.
        Para cada ponto de referência, encontramos seu vizinho mais próximo (no espaço)
        e acompanhamos a evolução da distância entre eles ao longo do tempo.
        A curva média do logaritmo das distâncias em função do tempo é ajustada por uma reta
        cujo coeficiente angular é o maior expoente de Lyapunov λ.

    Para que serve:
        - Detectar regimes de caos determinístico em mercados financeiros.
        - Quantificar a previsibilidade do sistema: λ > 0 indica caos e baixa previsibilidade
          de longo prazo; λ ≈ 0 sugere dinâmica periódica; λ < 0 indica estabilidade.
        - Auxiliar na gestão de risco: valores positivos elevados podem sinalizar períodos
          de alta instabilidade e menor confiabilidade de modelos preditivos.

    Interpretação dos valores gerados:
        - λ > 0.0   : Regime caótico – a previsibilidade é limitada a poucos passos à frente.
        - λ ≈ 0.0   : Comportamento periódico ou estacionário – as trajetórias mantêm
                       distâncias constantes.
        - λ < 0.0   : Sistema estável – as trajetórias convergem para um atrator.
        - NaN       : Não foi possível calcular (janela muito curta, dados insuficientes).

    Parâmetros:
        window (int): Tamanho da janela móvel (rolling window) para estimação.
        m (int): Dimensão de embedding (número de defasagens usadas na reconstrução).
        tau (int): Atraso temporal entre os elementos do vetor de embedding.
        min_t (int): Número mínimo de passos de tempo considerados no ajuste linear.
        max_t (int): Número máximo de passos a serem acompanhados.
        price_col (str): Nome da coluna de preços no DataFrame (padrão 'close').
    """
    def __init__(self, asset=None, timeframe=None, **params):
        super().__init__(asset, timeframe, **params)
        self.name = "lyapunov_exp"
        self.window = int(params.get('window', 500))
        self.m = int(params.get('m', 5))
        self.tau = int(params.get('tau', 1))
        self.min_t = int(params.get('min_t', 5))
        self.max_t = int(params.get('max_t', 30))
        self.price_col = params.get('price_col', 'close')

    def _calculate_logic(self, df: pl.DataFrame, **kwargs) -> pl.Series:
        # Obtém preços e calcula retornos logarítmicos
        prices = df[self.price_col].to_numpy()
        returns = np.diff(np.log(prices))
        # Adiciona NaN no início para manter comprimento original
        returns = np.concatenate(([np.nan], returns))
        ret_series = pl.Series(returns)

        n = len(ret_series)
        lyap_series = np.full(n, np.nan)

        for i in range(self.window, n):
            y = ret_series[i - self.window : i].to_numpy()
            # Remove possíveis NaNs (caso haja falhas nos dados)
            y_clean = y[~np.isnan(y)]
            if len(y_clean) < self.window * 0.8:  # menos de 80% de dados válidos
                lyap_series[i] = np.nan
                continue

            try:
                lyap = self._compute_lyapunov(y_clean)
                lyap_series[i] = lyap
            except Exception:
                lyap_series[i] = np.nan

        return pl.Series(lyap_series)

    def _compute_lyapunov(self, series: np.ndarray) -> float:
        """
        Estima o maior expoente de Lyapunov para uma série temporal unidimensional.
        Algoritmo baseado em Rosenstein et al. (1993).
        """
        N = len(series)
        # 1. Reconstrução do espaço de estados (embedding)
        if N < (self.m - 1) * self.tau + 1:
            raise ValueError("Série muito curta para a dimensão de embedding escolhida.")

        Y = self._embed(series, self.m, self.tau)
        n_vec = len(Y)

        # 2. Encontrar vizinhos mais próximos (excluindo vizinhos temporais)
        dist_matrix = cdist(Y, Y)
        neighbors = []
        for i in range(n_vec):
            # Ordena por distância (a menor é ela mesma, então pulamos)
            order = np.argsort(dist_matrix[i])
            for j in order[1:]:  # começa do segundo mais próximo
                if abs(i - j) > self.m:  # evita vizinhos muito próximos no tempo
                    neighbors.append((i, j))
                    break

        # 3. Acompanhar a divergência até max_t
        max_t = min(self.max_t, n_vec - max(i for i, _ in neighbors))
        if max_t < self.min_t:
            return np.nan

        div_avg = np.zeros(max_t)
        count = np.zeros(max_t)

        for t in range(max_t):
            for i, j in neighbors:
                if i + t < n_vec and j + t < n_vec:
                    d = np.linalg.norm(Y[i + t] - Y[j + t])
                    if d > 1e-10:
                        div_avg[t] += np.log(d)
                        count[t] += 1
            if count[t] > 0:
                div_avg[t] /= count[t]

        # 4. Ajuste linear nos primeiros passos (antes da saturação)
        # Usamos os pontos de min_t até max_t (ou até onde count > 0)
        valid = np.where(count > 0)[0]
        if len(valid) < self.min_t:
            return np.nan

        t_vals = valid[:self.min_t]  # pode ajustar o intervalo conforme necessidade
        y_vals = div_avg[t_vals]
        coeffs = np.polyfit(t_vals, y_vals, 1)
        lambda_max = coeffs[0]  # inclinação
        return lambda_max

    @staticmethod
    def _embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
        """Reconstrução do espaço de estados por atraso temporal."""
        n = len(x) - (m - 1) * tau
        if n <= 0:
            raise ValueError("Parâmetros de embedding incompatíveis com o tamanho da série.")
        result = np.zeros((n, m))
        for i in range(m):
            result[:, i] = x[i * tau : i * tau + n]
        return result