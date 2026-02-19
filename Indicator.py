import polars as pl
import itertools
import json

class Indicator:
    def __init__(self, asset: str = None, timeframe: str = None, **params):
        self.asset = asset
        self.timeframe = timeframe
        self.params = params
        #self.name = self.__class__.__name__.lower()

    def calculate(self, df: pl.DataFrame, param_set_dict: dict = None, ind_name: str = None): 
        """
        Resolve as variáveis do indicador (otimização) antes de chamar a lógica real.
        Recebe e deve retornar objetos do Polars.
        """
        # 1. RESOLUÇÃO DE PARÂMETROS (Mapeia strings para valores da otimização)
        effective_params = self.params.copy()
        if param_set_dict:
            for k, v in effective_params.items():
                if isinstance(v, str) and v in param_set_dict:
                    effective_params[k] = param_set_dict[v]
        
        effective_params['ind_name'] = ind_name if ind_name else self.name
        result = self._calculate_logic(df, **effective_params)

        if not isinstance(result, (pl.Series, pl.DataFrame)) and result is not None:
            # Usa o nome injetado para nomear a Series caso venha "pura"
            return pl.Series(effective_params['ind_name'], result)
            
        return result

    def _calculate_logic(self, df: pl.DataFrame, **kwargs):
        """
        Este método deve ser sobrescrito em cada indicador (ex: MA.py, RSI.py).
        Deve conter apenas cálculos usando expressões do Polars.
        """
        raise NotImplementedError("Subclasses must implement _calculate_logic method.")