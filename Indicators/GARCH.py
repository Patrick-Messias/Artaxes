import pandas as pd, numpy as np
from arch import arch_model
from typing import Optional, Callable, Literal
from tqdm import tqdm
import warnings, sys
warnings.filterwarnings("ignore")
sys.path.append(f'C:\\Users\\Patrick\\Desktop\\ART_Backtesting_Platform\\Backend')
from Indicator import Indicator

# =========================
# Defaults para cada modelo
# =========================
MEAN_DEFAULTS = {
    'Zero': None,
    'Constant': None,
    'AR': 1,
    'ARX': 1,
    'HAR': [1, 5, 22]
}

VOL_DEFAULTS = {
    'Arch': {'p': 1, 'q': 0},
    'Garch': {'p': 1, 'q': 1},
    'Figarch': {'p': 1, 'q': 1},
    'GJR': {'p': 1, 'o': 1, 'q': 1}
}

class GARCH(Indicator):
    def __init__(
        self,
        asset: str = None,
        timeframe: str = None,
        window: float = 252,
        mean_model: Literal['Zero', 'Constant', 'AR', 'ARX', 'HAR'] = 'Zero',
        vol_model: Literal['Arch', 'Garch', 'Figarch', 'GJR'] = 'Garch',
        dist: Literal['normal', 'gaussian', 'ged', 'generalized error', 'skewstudent', 'skewt', 'studentst', 't'] = 'normal',
        rolling_mode: Literal['rolling', 'anchored'] = 'rolling',
        price_col_func: Optional[Callable[[pd.DataFrame], pd.Series]] = lambda df_: np.log(df_['close'] / df_['close'].shift(1)).replace([np.inf, -np.inf], np.nan).ffill(),
        verbose: bool = True
    ):
        self.asset = asset
        self.timeframe = timeframe
        self.window = window
        self.mean_model = mean_model
        self.vol_model = vol_model
        self.dist = dist
        self.rolling_mode = rolling_mode
        self.price_col_func = price_col_func
        self.verbose = verbose

    # =========================
    # Função principal de ajuste e previsão
    # =========================
    def calculate(self, df: pd.DataFrame):
        self.df = df.copy().reset_index(drop=True)

        # calcular returns
        self.df['ret'] = self.price_col_func(self.df)

        if self.window == 1.0:
            train = self.df['ret'].dropna()
            vol_type = self.vol_model if self.vol_model in ['Arch', 'Figarch'] else 'GARCH'
            arch_kwargs = VOL_DEFAULTS.get(self.vol_model, {})
            am = arch_model(train, mean=self.mean_model, vol=vol_type, dist=self.dist, **arch_kwargs)
            res = am.fit(disp="off")
            fc = res.forecast(horizon=1)
            next_forecast = np.sqrt(fc.variance.values[-1, 0])
            if self.verbose:
                print(f"Forecast próximo período: {next_forecast:.6f}")
            return next_forecast
        
        self.df['rv'] = np.log(self.df['high'] / self.df['low']).replace([np.inf, -np.inf], np.nan).ffill()
        self.df.dropna(inplace=True)

        n = len(self.df)
        self.predictions = [np.nan] * n

        # Definir índices de treino/teste
        if 0 < self.window < 1:
            split_idx = int(n * self.window)
            test_idx = range(split_idx, n)
        else:
            split_idx = int(self.window)
            test_idx = range(split_idx, n)

        # =========================
        # Treinamento e previsão rolling
        # =========================
        for i in tqdm(test_idx):
            train = self.df['ret'][max(0, i - split_idx):i] if self.rolling_mode == 'rolling' else self.df['ret'][:i]
            
            try:
                arch_kwargs = VOL_DEFAULTS.get(self.vol_model, {})
                if self.mean_model in MEAN_DEFAULTS:
                    arch_kwargs['lags'] = MEAN_DEFAULTS[self.mean_model]

                vol_type = self.vol_model if self.vol_model in ['Arch', 'Figarch'] else 'GARCH'
                am = arch_model(train, mean=self.mean_model, vol=vol_type, dist=self.dist, **arch_kwargs)
                res = am.fit(disp="off")
                fc = res.forecast(horizon=1)
                pred = np.sqrt(fc.variance.values[-1, 0])
                self.predictions[i] = pred if not np.isnan(pred) else train.std()
            except Exception:
                self.predictions[i] = train.std()

        self.df['prediction'] = self.predictions

        # =========================
        # Métricas simples
        # =========================
        def _rmse(y_true, y_pred):
            mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
            return np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2))

        def _mae(y_true, y_pred):
            mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
            return np.mean(np.abs(y_true[mask] - y_pred[mask]))

        valid_mask = ~self.df['prediction'].isna()
        mae = _mae(self.df.loc[valid_mask, 'rv'], self.df.loc[valid_mask, 'prediction'])
        rmse = _rmse(self.df.loc[valid_mask, 'rv'], self.df.loc[valid_mask, 'prediction'])

        if self.verbose:
            print(f"MAE: {mae:.6f}, RMSE: {rmse:.6f}")

        return self.df

# =========================
# Exemplo de uso
# =========================
if __name__ == "__main__":
    df = pd.read_excel(r'C:\Users\Patrick\Desktop\Artaxes Portfolio\MAIN\MT5_Dados\WIN$_D1.xlsx')

    col = lambda df_: np.log(df_['close'] / df_['close'].shift(1)).replace([np.inf, -np.inf], np.nan).ffill()

    model = GARCH(window=252, mean_model='Zero', vol_model='Garch', dist='normal', price_col_func=col)
    df_pred = model.calculate(df)
    print(df_pred)





