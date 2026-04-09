
import polars as pl
from typing import List, Dict, Any, Union

class Indicator:
    # Base class for all technical indicators
    # Focuses on generating Polars Expressions for high-performance batch processing

    def __init__(self, asset: str = None, timeframe: str = None, **params):
        self.asset = asset
        self.timeframe = timeframe
        self.params = params
        self.name = self.__class__.__name__.lower()

    def _get_expr(self, **kwargs) -> Union[pl.Expr, List[pl.Expr]]:
        """Must be implemented by subclasses. Returns one or more pl.Expr."""
        raise NotImplementedError("Subclasses must implement _get_expr")

    def get_expression(self, param_set: Dict[str, Any]=None, alias: str=None) -> pl.Expr:
        # Resolves parameters and returns the final aliased expression
        effective_params = self.params.copy()
        if param_set:
            effective_params.update(param_set)
        
        exprs = self._get_expr(**effective_params)
        
        # Suffix for unique identification: e.g., _21_close
        suffix = "_".join([str(v) for k, v in effective_params.items() if k not in ['price_col', 'high_col', 'low_col']])
        
        if isinstance(exprs, list): # If multiple columns, alias each one
            return [e.alias(f"{e.meta.output_name()}_{suffix}") for e in exprs]
        return exprs.alias(f"{self.name}_{suffix}")
    
    def batch_expressions(self, param_list: List[Dict[str, Any]]) -> List[pl.Expr]:
        """Flattens all expressions from multiple parameter sets into one list."""
        all_exprs = []
        for ps in param_list:
            res = self.get_expression(ps)
            if isinstance(res, list):
                all_exprs.extend(res)
            else:
                all_exprs.append(res)
        return all_exprs



"""
# Refazer para:
# calculate_single: Lógica pura, com 1 set de dado
# calculate: Itera sobre data com rolling e chama calculate_single
# calculate_param_sets: Itera sobre cada parset e chama calculate

class Indicator:
    def __init__(self, asset: str = None, timeframe: str = None, when: Literal["pre", "live"]="pre", **params):
        self.asset = asset
        self.timeframe = timeframe
        self.when = when
        self.params = params
        #self.name = self.__class__.__name__.lower()

    def calculate(self, df, param_set_dict: dict = None, ind_name: str = None): 
        # Resolve as variáveis do indicador (otimização) antes de chamar a lógica real.
        # Recebe e deve retornar objetos do Polars.
       
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
    
    

    def _calculate_logic(self, df, **kwargs):
        # Este método deve ser sobrescrito em cada indicador (ex: MA.py, RSI.py).
        # Deve conter apenas cálculos usando expressões do Polars.

        raise NotImplementedError("Subclasses must implement _calculate_logic method.")

"""


