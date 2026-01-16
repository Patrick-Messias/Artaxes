import pandas as pd
from Indicator import Indicator

class MA(Indicator):
    """
    Moving Average indicator
    
    Parameters:
        window: int, list or range - MA period (default: 21)
        ma_type: str or list - 'sma' or 'ema' (default: 'sma')  
        price_col: str - Price column (default: 'close')
    """
    
    def __init__(self, asset=None, timeframe=None, **params):
        super().__init__(asset, timeframe, **params)
        self.params = params  # Define self.params explicitamente para garantir atualização
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        window = self.params.get('window', 21)
        ma_type = self.params.get('ma_type', 'sma')
        price_col = self.params.get('price_col', 'close')

        if ma_type == 'sma':
            return df[price_col].rolling(window=window, min_periods=window).mean()
        elif ma_type == 'ema':
            return df[price_col].ewm(span=window, adjust=False).mean()
        else:
            raise ValueError(f"Unsupported MA type: {ma_type}")
        

        

# class MA(Indicator):
#     def __init__(self, asset, timeframe: str, window: int = 20, type: str = 'sma', price_col: str = 'close'):
#         super().__init__(asset, timeframe)
#         self.window = window
#         self.type = type 
#         self.price_col = price_col

#     def calculate(self, df: pd.DataFrame) -> pd.Series:
#         if self.type == 'sma':
#             return df[self.price_col].rolling(window=self.window, min_periods=self.window).mean()
#         elif self.type == 'ema':
#             return df[self.price_col].ewm(span=self.window, adjust=False).mean()
#         else:
#             raise ValueError(f"Unsupported MA type: {self.type}")




