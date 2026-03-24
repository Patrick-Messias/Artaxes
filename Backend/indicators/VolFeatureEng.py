import numpy as np
import pandas as pd

class VolFeatureEng:
    """
    Volatility Feature Engineering for HAR, GARCH, SV, HMM and ML models.

    Inputs:
        df: DataFrame with columns ['open', 'high', 'low', 'close'] (and optional 'rv')
        realized_col: name of intraday realized volatility column if available
        freq: daily frequency (default='D') for HAR-type aggregations
    Output:
        A DataFrame with:
            - range-based vol estimators
            - HAR-style lag features
            - returns, overnight jumps, calendar dummies
    """

    def __init__(self, df: pd.DataFrame, realized_col: str = None, freq: str = 'D', params: dict = {'har_period_s': 5, 'har_period_m': 21, 'vol_yz_period_m': 21}):
        self.df = df.copy()
        self.realized_col = realized_col
        self.freq = freq
        self._prepare_df()
        self.params = params

    # ------------------------------------------------------------------
    # 1. Preparation
    def _prepare_df(self):
        self.df = self.df.sort_index()
        self.df['ret'] = np.log(self.df['close'] / self.df['close'].shift(1))
        self.df['overnight'] = np.log(self.df['open'] / self.df['close'].shift(1))
        self.df['abs_ret'] = self.df['ret'].abs()

    # ------------------------------------------------------------------
    # 2. Range-based volatility estimators
    def parkinson(self):
        return 0.5 * (np.log(self.df['high'] / self.df['low']))**2

    def garman_klass(self):
        hl = np.log(self.df['high'] / self.df['low'])
        co = np.log(self.df['close'] / self.df['open'])
        return 0.5 * hl**2 - (2*np.log(2) - 1) * co**2

    def rogers_satchell(self):
        return (np.log(self.df['high'] / self.df['close']) *
                np.log(self.df['high'] / self.df['open']) +
                np.log(self.df['low'] / self.df['close']) *
                np.log(self.df['low'] / self.df['open']))

    def yang_zhang(self):
        o, h, l, c = self.df['open'], self.df['high'], self.df['low'], self.df['close']
        k = 0.34 / (1.34 + (len(self.df) / (len(self.df) - 1)))
        log_oc = np.log(o / self.df['close'].shift(1))
        log_co = np.log(c / o)
        rs = np.log(h / l)
        return (log_oc**2 +
                k * log_co**2 +
                (1 - k) * (rs**2))

    # ------------------------------------------------------------------
    # 3. Generate all vol estimators
    def compute_all_vols(self):
        self.df['vol_parkinson'] = self.parkinson()
        self.df['vol_gk'] = self.garman_klass()
        self.df['vol_rs'] = self.rogers_satchell()
        self.df['vol_yz'] = self.yang_zhang()
        if self.realized_col and self.realized_col in self.df.columns:
            self.df['vol_rv'] = self.df[self.realized_col]
        return self.df

    # ------------------------------------------------------------------
    # 4. HAR-style features (daily/weekly/monthly realized vol)
    def make_har_features(self, base_col='vol_yz'):
        self.df[f'{base_col}_daily'] = self.df[base_col].shift(1)
        self.df[f'{base_col}_weekly'] = self.df[base_col].shift(1).rolling(self.params['har_period_s']).mean()
        self.df[f'{base_col}_monthly'] = self.df[base_col].shift(1).rolling(self.params['har_period_m']).mean()
        return self.df

    # ------------------------------------------------------------------
    # 5. Additional contextual features
    def add_context_features(self):
        # Garante que o índice é datetime
        if not isinstance(self.df.index, pd.DatetimeIndex):
            if 'datetime' in self.df.columns:
                self.df['datetime'] = pd.to_datetime(self.df['datetime'])
                self.df.set_index('datetime', inplace=True)
            else:
                raise ValueError("O DataFrame precisa ter uma coluna 'datetime' para criar as features de contexto.")

        self.df[f"vol_mean_{self.params['vol_yz_period_m']}"] = self.df['vol_yz'].rolling(self.params['vol_yz_period_m']).mean()
        self.df['jump_abs'] = self.df['overnight'].abs()
        self.df['dow'] = self.df.index.weekday
        self.df['month'] = self.df.index.month
        self.df['regime_dummy'] = (self.df['vol_yz'] > self.df[f"vol_mean_{self.params['vol_yz_period_m']}"]).astype(int)
        return self.df

    # ------------------------------------------------------------------
    # 6. Final method: build everything
    def build_all(self):
        self.compute_all_vols()
        self.make_har_features(base_col='vol_yz')
        self.add_context_features()
        self.df.dropna(inplace=True)
        return self.df

    # ------------------------------------------------------------------
    # 7. Output feature columns ready for ML/HMM/SV/GARCH
    def get_feature_matrix(self):
        cols = [
            'vol_parkinson', 'vol_gk', 'vol_rs', 'vol_yz',
            'vol_yz_daily', 'vol_yz_weekly', 'vol_yz_monthly',
            'ret', 'abs_ret', 'overnight', 'jump_abs',
            'dow', 'month'
        ]
        cols = [c for c in cols if c in self.df.columns]
        X = self.df[cols].copy()
        y = (self.df['vol_yz'].shift(-1) >= self.df[f"vol_mean_{self.params['vol_yz_period_m']}"]).astype(int).dropna()
        X = X.iloc[:len(y)]
        return X, y
