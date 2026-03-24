import pandas as pd
import numpy as np
from typing import Optional, Callable, Literal
from tqdm import tqdm
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler

import warnings, sys
warnings.filterwarnings("ignore")
sys.path.append(f'C:\\Users\\Patrick\\Desktop\\ART_Backtesting_Platform\\Backend')
from Indicator import Indicator

# Para prever se o preço vai subir/descer
target_col = 'ret'  # ou pct_change
p, d, q = 1, 0, 1   # ARIMA em retornos (série estacionária)

# Para prever períodos de alta/baixa volatilidade
target_col = 'rv'   # log(high/low)
p, d, q = 1, 0, 1   # ARIMA em volatilidade

# Para prever níveis absolutos de preço
target_col = 'close'
p, d, q = 1, 1, 1   # ARIMA com diferenciação (d=1)

class ARIMAInd(Indicator):
    def __init__(
        self,
        asset: str = None,
        timeframe: str = None,
        target_col: str = 'ret',
        p: int = 1,
        d: int = 0,
        q: int = 1,
        window: int = 252,
        normalize_output: bool = False,
        scale: bool = False,
        forecast_steps: int = 1,
        max_iter: int = 50,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
        handle_errors: bool = True,
        verbose: bool = True
    ):
        super().__init__(asset, timeframe)
        self.target_col = target_col
        self.p, self.d, self.q = p, d, q
        self.window = window
        self.normalize_output = normalize_output
        self.scale = scale
        self.forecast_steps = forecast_steps
        self.max_iter = max_iter
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.handle_errors = handle_errors
        self.verbose = verbose
        
        self.scaler = None
        self.df = None
        self.predictions = None

    def _normalize_series(self, series: pd.Series) -> pd.Series:
        """Normaliza uma série entre -1 e 1"""
        if len(series) == 0:
            return series
        max_val = np.max(np.abs(series))
        if max_val == 0:
            return series
        return series / max_val

    def _scale_data(self, data: pd.Series) -> pd.Series:
        """Aplica scaling nos dados se necessário"""
        if not self.scale or len(data) < 2:
            return data
            
        try:
            self.scaler = StandardScaler()
            scaled_values = self.scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
            return pd.Series(scaled_values, index=data.index)
        except:
            return data

    def _fit_arima(self, train_data: pd.Series) -> Optional[ARIMA]:
        """Ajusta o modelo ARIMA aos dados de treino"""
        try:
            model = ARIMA(
                train_data,
                order=(self.p, self.d, self.q)
            )
            # API correta para statsmodels - removendo parâmetros não suportados
            fitted_model = model.fit()
            return fitted_model
        except Exception as e:
            if self.verbose and not self.handle_errors:
                print(f"Erro ao ajustar ARIMA: {e}")
            return None

    def _make_forecast(self, model: ARIMA) -> Optional[float]:
        """Faz previsão com o modelo ARIMA ajustado"""
        try:
            forecast = model.forecast(steps=self.forecast_steps)
            return float(forecast.iloc[-1]) if hasattr(forecast, 'iloc') else float(forecast[-1])
        except Exception as e:
            if self.verbose and not self.handle_errors:
                print(f"Erro na previsão: {e}")
            return None

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula as previsões ARIMA para o DataFrame fornecido.
        """
        self.df = df.copy().reset_index(drop=True)

        if len(self.df) < self.window:
            raise ValueError(f"Not enough data with window size")

        # Verificar se a coluna alvo existe
        if self.target_col not in self.df.columns:
            if self.target_col == 'ret':
                self.df['ret'] = self.df['close'].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
            elif self.target_col == 'rv':
                self.df['rv'] = np.log(df['high'] / df['low']).replace([np.inf, -np.inf], np.nan).fillna(0)
            else:
                raise ValueError(f"Coluna '{self.target_col}' não encontrada no DataFrame")
        
        # Usar a coluna target diretamente
        target_series = self.df[self.target_col]
        n = len(target_series)
        
        # Criar array de previsões
        self.predictions = np.full(n, np.nan)
        
        # =========================
        # Caso 1: Previsão única (window = 1)
        # =========================
        if self.window == 1:
            # if n < self.min_periods:
            #     if self.verbose:
            #         print(f"Dados insuficientes: {n} < {self.min_periods}")
            #     self.df['arima_prediction'] = self.predictions
            #     return self.df
                
            # Preparar dados
            # train_data = target_series.dropna()
            # if len(train_data) < self.min_periods:
            #     if self.verbose:
            #         print(f"Dados válidos insuficientes após remover NaN: {len(train_data)} < {self.min_periods}")
            #     self.df['arima_prediction'] = self.predictions
            #     return self.df
            
            if self.scale:
                train_data = self._scale_data(train_data)
            
            # Ajustar modelo e fazer previsão
            model = self._fit_arima(train_data)
            if model is not None:
                forecast = self._make_forecast(model)
                if forecast is not None:
                    if self.normalize_output:
                        forecast = self._normalize_series(pd.Series([forecast])).iloc[0]
                    
                    if self.verbose:
                        print(f"Previsão ARIMA({self.p},{self.d},{self.q}): {forecast:.6f}")
                    
                    # Preencher a última posição com a previsão
                    self.predictions[-1] = forecast
            
            self.df['arima_prediction'] = self.predictions
            return self.df
        
        # =========================
        # Caso 2: Janela rolante (window > 1)
        # =========================
        
        # Definir índices de início
        start = self.window-1 if self.window > 1 else 0
        start_idx = max(start, self.window)
        
        if start_idx >= n:
            if self.verbose:
                print(f"Window muito grande: {self.window} > {n}")
            self.df['arima_prediction'] = self.predictions
            return self.df
        
        # Índices para previsão
        test_indices = range(start_idx, n)
        
        if self.verbose:
            print(f"Calculando {len(test_indices)} previsões ARIMA...")
            test_indices = tqdm(test_indices)
        
        successful_fits = 0
        failed_fits = 0
        
        for i in test_indices:
            # Definir janela de treino
            start_train = max(0, i - self.window)
            train_window = target_series.iloc[start_train:i]
            
            # Remover NaNs da janela de treino
            train_data = train_window.dropna()
            
            # if len(train_data) < self.min_periods:
            #     continue
            
            # Aplicar scaling se necessário
            if self.scale:
                train_data = self._scale_data(train_data)
            
            # Ajustar modelo ARIMA
            model = self._fit_arima(train_data)
            
            if model is not None:
                successful_fits += 1
                # Fazer previsão
                forecast = self._make_forecast(model)
                
                if forecast is not None:
                    # Aplicar normalização se necessário
                    if self.normalize_output:
                        forecast = self._normalize_series(pd.Series([forecast])).iloc[0]
                    
                    self.predictions[i] = forecast
            else:
                failed_fits += 1
        
        if self.verbose:
            print(f"Modelos ajustados com sucesso: {successful_fits}, Falhas: {failed_fits}")
        
        # Adicionar previsões ao DataFrame
        self.df['arima_prediction'] = self.predictions
        
        # =========================
        # Métricas (se dados suficientes)
        # =========================
        if self.verbose:
            valid_predictions = self.df['arima_prediction'].notna()
            valid_count = valid_predictions.sum()
            
            if valid_count > 0:
                print(f"ARIMA({self.p},{self.d},{self.q}) - Previsões válidas: {valid_count}/{n}")
                
                # Calcular métricas se tivermos valores reais para comparar
                if valid_count > 1:
                    actual_values = self.df.loc[valid_predictions, self.target_col]
                    predicted_values = self.df.loc[valid_predictions, 'arima_prediction']
                    
                    mae = np.mean(np.abs(actual_values - predicted_values))
                    rmse = np.sqrt(np.mean((actual_values - predicted_values) ** 2))
                    
                    print(f"MAE: {mae:.6f}, RMSE: {rmse:.6f}")
            else:
                print("Nenhuma previsão válida gerada")
        
        return self.df

    def get_summary(self) -> dict:
        """Retorna um resumo das previsões"""
        if self.df is None or 'arima_prediction' not in self.df.columns:
            return {}
        
        valid_preds = self.df['arima_prediction'].dropna()
        
        if len(valid_preds) == 0:
            return {}
        
        return {
            'model': f'ARIMA({self.p},{self.d},{self.q})',
            'total_predictions': len(valid_preds),
            'mean_prediction': float(valid_preds.mean()),
            'std_prediction': float(valid_preds.std()),
            'min_prediction': float(valid_preds.min()),
            'max_prediction': float(valid_preds.max())
        }


# =========================
# Exemplo de uso CORRIGIDO
# =========================
if __name__ == "__main__":
    df = pd.read_excel(r'C:\Users\Patrick\Desktop\Artaxes Portfolio\MAIN\MT5_Dados\WIN$_D1.xlsx')

    # Momentum = força do movimento recente
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1    # 5 períados
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1  # 10 períodos
    df['momentum_21'] = df['close'] / df['close'].shift(21) - 1  # 1 mês
    df['drift'] = df['close'].pct_change().fillna(0).rolling(window=21).mean().fillna(0)

    returns = df['close'].pct_change()
    df['drift_sharpe'] = returns.rolling(21).mean() / returns.rolling(21).std()

    df['velocity'] = df['close'].pct_change().diff()

    # Testar o indicador - usar window menor para teste
    arima_ind = ARIMAInd(
        target_col='drift',  # Usar preços
        p=1, d=0, q=1,      # AR(1) simples para teste
        window=21,          # Janela menor para teste
        normalize_output=False,
        scale=False,
        forecast_steps=1,
        verbose=True,
        handle_errors=True   # Mudar para True para continuar mesmo com erros
    )
    
    result_df = arima_ind.calculate(df)
    print(f"\nPrimeiras previsões:\n{result_df[['date', 'close', 'arima_prediction']].head(10)}")
    print(f"\nÚltimas previsões (com valores):")
    
    # Mostrar apenas as previsões não nulas
    print(result_df)
    


    import matplotlib.pyplot as plt

    # Plot básico
    plt.figure(figsize=(12, 6))
    plt.style.use('dark_background')
    plt.plot(result_df['date'], result_df['drift'], linewidth=1, alpha=0.7)
    plt.plot(result_df['date'], result_df['arima_prediction'], label='ARIMA', linewidth=1, alpha=0.7)
    plt.title(f'ARIMA({arima_ind.p},{arima_ind.d},{arima_ind.q}) Predictions vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


