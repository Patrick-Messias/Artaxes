import polars as pl
import numpy as np
from scipy.optimize import minimize
from Indicator import Indicator

class QbSD(Indicator):
    """
    Indicador QbSD para previsão de Value-at-Risk (VaR) com modelo de escala baseado em quantis.a
    Inspirado pelo paper: "Quantile-based modeling of scale dynamics in financial - Xiaochun Liu and Richard Luger"
    
    Parâmetros:
        alpha (float): Nível de probabilidade do VaR (ex: 0.01 para 1%).
        p_list (list): Níveis de quantil usados para construir a escala (ex: [0.05,0.10,0.15,0.20,0.25]).
        window (int): Tamanho da janela móvel (rolling window) para estimação.
        model_type (str): 'gAS' (assimétrico, padrão) ou 'gSAV' (simétrico).
        price_col (str): Nome da coluna de preços no DataFrame (padrão 'close').
    """
    def __init__(self, asset=None, timeframe=None, **params):
        super().__init__(asset, timeframe, **params)
        self.name = "qbsd"
        self.alpha = params.get('alpha', 0.01)
        self.p_list = params.get('p_list', [0.05, 0.10, 0.15, 0.20, 0.25])
        self.window = int(params.get('window', 1250))
        self.model_type = params.get('model_type', 'gAS')
        self.price_col = params.get('price_col', 'close')

    def _calculate_logic(self, df: pl.DataFrame, **kwargs) -> pl.Series:
        # Obtém a série de preços e calcula retornos logarítmicos
        prices = df[self.price_col].to_numpy()
        returns = np.diff(np.log(prices))
        # Adiciona NaN no início para manter o mesmo comprimento do DataFrame original
        returns = np.concatenate(([np.nan], returns))
        ret_series = pl.Series(returns)

        n = len(ret_series)
        var_forecast = np.full(n, np.nan)

        # Dicionário para armazenar os últimos parâmetros estimados para cada p (acelera convergência)
        last_params_dict = {p: None for p in self.p_list}

        for i in range(self.window, n):
            # Janela de retornos: índices i-window até i-1 (exclui o retorno atual)
            y = ret_series[i - self.window : i].to_numpy()

            # Previsão VaR para o período i (baseada na janela que termina em i-1)
            forecast, new_params_dict = self._forecast_one(y, last_params_dict)
            var_forecast[i] = forecast

            # Atualiza os parâmetros para a próxima iteração
            last_params_dict = new_params_dict

        return pl.Series(var_forecast)

    def _forecast_one(self, y, last_params_dict):
        """
        Estima o modelo para cada p e retorna a previsão VaR agregada (média).
        Também retorna um dicionário com os novos parâmetros para cada p.
        """
        forecasts = []
        new_params = {}

        for p in self.p_list:
            if self.model_type == 'gAS':
                params, Q_p, Q_1mp = self._estimate_gAS(y, p, last_params_dict.get(p))
            else:  # gSAV
                params, Q_p, Q_1mp = self._estimate_gSAV(y, p, last_params_dict.get(p))

            new_params[p] = params

            # Série da escala: s_t = Q_{1-p} - Q_p
            s = Q_1mp - Q_p
            # Resíduos padronizados
            epsilon = y / s
            # Quantil empírico dos resíduos no nível alpha (cauda esquerda)
            var_epsilon = np.percentile(epsilon, self.alpha * 100)

            # Previsão one-step-ahead da escala
            y_last = y[-1]
            Q_p_last = Q_p[-1]
            Q_1mp_last = Q_1mp[-1]
            s_last = s[-1]
            eps_last = y_last / s_last

            if self.model_type == 'gAS':
                omega_p, omega_1mp, beta, gamma_plus, gamma_minus = params
                gamma_term = gamma_plus if y_last > 0 else gamma_minus
            else:
                omega_p, omega_1mp, beta, gamma = params
                gamma_term = gamma

            Q_p_next = omega_p + (beta + gamma_term * abs(eps_last)) * Q_p_last
            Q_1mp_next = omega_1mp + (beta + gamma_term * abs(eps_last)) * Q_1mp_last
            s_next = Q_1mp_next - Q_p_next

            # Previsão VaR para o próximo período
            var_next = s_next * var_epsilon
            forecasts.append(var_next)

        # Agregação pela média sobre os diferentes p
        return np.mean(forecasts), new_params

    def _estimate_gAS(self, y, p, initial_params=None):
        """
        Estima o modelo gAS (assimétrico) para um dado nível p.
        Retorna os parâmetros otimizados e as séries Q_p e Q_{1-p}.
        """
        T = len(y)
        # Quantis amostrais para inicialização
        q_p_init = np.percentile(y, p * 100)
        q_1mp_init = np.percentile(y, (1 - p) * 100)

        # Chute inicial
        if initial_params is not None and len(initial_params) == 5:
            x0 = initial_params
        else:
            x0 = [q_p_init, q_1mp_init, 0.5, 0.1, 0.1]   # omega_p, omega_1mp, beta, gamma_plus, gamma_minus

        def loss(params):
            omega_p, omega_1mp, beta, gamma_plus, gamma_minus = params
            Q_p = np.zeros(T)
            Q_1mp = np.zeros(T)
            Q_p[0] = q_p_init
            Q_1mp[0] = q_1mp_init
            total_loss = 0.0

            for t in range(1, T):
                s_prev = Q_1mp[t-1] - Q_p[t-1]
                if s_prev <= 0:
                    return 1e10  # penalidade para evitar escala não positiva
                eps_prev = y[t-1] / s_prev
                gamma_term = gamma_plus if y[t-1] > 0 else gamma_minus
                Q_p[t] = omega_p + (beta + gamma_term * abs(eps_prev)) * Q_p[t-1]
                Q_1mp[t] = omega_1mp + (beta + gamma_term * abs(eps_prev)) * Q_1mp[t-1]

                loss_t = self._check_loss(y[t], Q_p[t], p) + self._check_loss(y[t], Q_1mp[t], 1-p)
                total_loss += loss_t
            return total_loss

        # Restrições: omega_1mp >= omega_p, beta >=0, gamma_plus >=0, gamma_minus >=0
        cons = (
            {'type': 'ineq', 'fun': lambda params: params[1] - params[0]},
            {'type': 'ineq', 'fun': lambda params: params[2]},
            {'type': 'ineq', 'fun': lambda params: params[3]},
            {'type': 'ineq', 'fun': lambda params: params[4]}
        )

        res = minimize(loss, x0, constraints=cons, method='SLSQP',
                       options={'maxiter': 500, 'ftol': 1e-6})

        if not res.success:
            # Em caso de falha, usa os valores iniciais
            params = x0
        else:
            params = res.x

        # Recalcula as séries Q com os parâmetros finais
        omega_p, omega_1mp, beta, gamma_plus, gamma_minus = params
        Q_p = np.zeros(T)
        Q_1mp = np.zeros(T)
        Q_p[0] = q_p_init
        Q_1mp[0] = q_1mp_init
        for t in range(1, T):
            s_prev = Q_1mp[t-1] - Q_p[t-1]
            eps_prev = y[t-1] / s_prev
            gamma_term = gamma_plus if y[t-1] > 0 else gamma_minus
            Q_p[t] = omega_p + (beta + gamma_term * abs(eps_prev)) * Q_p[t-1]
            Q_1mp[t] = omega_1mp + (beta + gamma_term * abs(eps_prev)) * Q_1mp[t-1]

        return params, Q_p, Q_1mp

    def _estimate_gSAV(self, y, p, initial_params=None):
        """
        Estima o modelo gSAV (simétrico) para um dado nível p.
        Retorna os parâmetros otimizados e as séries Q_p e Q_{1-p}.
        """
        T = len(y)
        q_p_init = np.percentile(y, p * 100)
        q_1mp_init = np.percentile(y, (1 - p) * 100)

        if initial_params is not None and len(initial_params) == 4:
            x0 = initial_params
        else:
            x0 = [q_p_init, q_1mp_init, 0.5, 0.1]   # omega_p, omega_1mp, beta, gamma

        def loss(params):
            omega_p, omega_1mp, beta, gamma = params
            Q_p = np.zeros(T)
            Q_1mp = np.zeros(T)
            Q_p[0] = q_p_init
            Q_1mp[0] = q_1mp_init
            total_loss = 0.0

            for t in range(1, T):
                s_prev = Q_1mp[t-1] - Q_p[t-1]
                if s_prev <= 0:
                    return 1e10
                eps_prev = y[t-1] / s_prev
                Q_p[t] = omega_p + (beta + gamma * abs(eps_prev)) * Q_p[t-1]
                Q_1mp[t] = omega_1mp + (beta + gamma * abs(eps_prev)) * Q_1mp[t-1]

                loss_t = self._check_loss(y[t], Q_p[t], p) + self._check_loss(y[t], Q_1mp[t], 1-p)
                total_loss += loss_t
            return total_loss

        cons = (
            {'type': 'ineq', 'fun': lambda params: params[1] - params[0]},
            {'type': 'ineq', 'fun': lambda params: params[2]},
            {'type': 'ineq', 'fun': lambda params: params[3]}
        )

        res = minimize(loss, x0, constraints=cons, method='SLSQP',
                       options={'maxiter': 500, 'ftol': 1e-6})

        if not res.success:
            params = x0
        else:
            params = res.x

        omega_p, omega_1mp, beta, gamma = params
        Q_p = np.zeros(T)
        Q_1mp = np.zeros(T)
        Q_p[0] = q_p_init
        Q_1mp[0] = q_1mp_init
        for t in range(1, T):
            s_prev = Q_1mp[t-1] - Q_p[t-1]
            eps_prev = y[t-1] / s_prev
            Q_p[t] = omega_p + (beta + gamma * abs(eps_prev)) * Q_p[t-1]
            Q_1mp[t] = omega_1mp + (beta + gamma * abs(eps_prev)) * Q_1mp[t-1]

        return params, Q_p, Q_1mp

    @staticmethod
    def _check_loss(u, q, tau):
        """Função check da regressão quantílica."""
        diff = u - q
        return diff * (tau - (diff < 0))