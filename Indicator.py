import pandas as pd, itertools, json

class Indicator:
    def __init__(self, asset: str=None, timeframe: str=None, **params):
        self.asset = asset
        self.timeframe = timeframe
        self.params = params
        self.name = self.__class__.__name__.lower()

    def calculate(self, df: pd.DataFrame, param_set_dict: dict = None): 
        # Solves indicator vars before calling real logic
        effective_params = self.params.copy()
        if param_set_dict:
            for k, v in effective_params.items():
                if isinstance(v, str) and v in param_set_dict:
                    effective_params[k] = param_set_dict[v]
        return self._calculate_logic(df, **effective_params)

    def _calculate_logic(self, df: pd.DataFrame, **kwargs):
        raise NotImplementedError("Subclasses must implement _calculate_logic method.")

    # 1. Agora deve retornar um dict com as keys do param_sets, sem recalcular desn, podendo ser chamado com param_set
    # def calculate_all_sets(self, df: pd.DataFrame, param_sets: dict = None, asset_name: str = None, timeframe: str = None, sep: str = '-') -> dict:
    #     """Calculate indicator outputs for a set of global param_sets.

    #     Behavior:
    #     - `param_sets` is a dict {global_ps_name: {param_name: value, ...}}.
    #     - For each global param_set we derive `effective_params` for this indicator
    #       (mapping names or substitutions from the indicator's `self.params`).
    #     - Deduplicate identical `effective_params` combinations and compute each
    #       unique indicator parameter set only once. Then assign the resulting
    #       DataFrame reference to every global_ps that maps to it.

    #     Returns structure: { asset_name: { timeframe: { global_param_set_name: DataFrame } } }
    #     """

    #     asset_key = asset_name or 'default_asset'
    #     tf_key = timeframe or getattr(self, 'timeframe', 'default_tf')
    #     results = {asset_key: {tf_key: {}}}

    #     # snapshot of defaults
    #     ind_defaults = getattr(self, 'params', {}) or {}

    #     # If no global param_sets provided, fallback to single current params
    #     if not param_sets:
    #         # try to call calculate with current params
    #         try:
    #             out = self.calculate(df, ind_defaults)
    #         except TypeError:
    #             backup = getattr(self, 'params', {}).copy()
    #             try:
    #                 self.params = ind_defaults
    #                 out = self.calculate(df)
    #             finally:
    #                 self.params = backup
    #         results[asset_key][tf_key]['default'] = out
    #         return results

    #     # helper: normalize value into hashable/serializable form
    #     def _norm(v):
    #         if isinstance(v, (list, tuple)):
    #             return tuple(v)
    #         if isinstance(v, dict):
    #             return json.dumps(v, sort_keys=True)
    #         return v

    #     # Build mapping: ind_key -> {'params': dict, 'global_ps_names': [..]}
    #     unique_map = {}

    #     for global_name, global_vals in param_sets.items():
    #         # build effective params for this indicator from global param set
    #         effective = {}
    #         # priority: if ind default value is a string referencing a key in global_vals
    #         for k, v in ind_defaults.items():
    #             if isinstance(v, str) and v in global_vals:
    #                 effective[k] = global_vals[v]
    #             elif k in global_vals:
    #                 effective[k] = global_vals[k]
    #             else:
    #                 effective[k] = v

    #         # also include any global_vals that match indicator params names
    #         for k, v in global_vals.items():
    #             if k not in effective and k in ind_defaults:
    #                 effective[k] = v

    #         # normalize into a key for deduplication
    #         ind_key = tuple(sorted((kk, _norm(vv)) for kk, vv in effective.items()))

    #         if ind_key not in unique_map:
    #             unique_map[ind_key] = {'params': effective, 'global_ps_names': []}
    #         unique_map[ind_key]['global_ps_names'].append(global_name)

    #     # compute each unique indicator param once
    #     for ind_key, info in unique_map.items():
    #         params_to_use = info['params']
    #         try:
    #             out = self.calculate(df, params_to_use)
    #         except TypeError:
    #             backup = getattr(self, 'params', {}).copy()
    #             try:
    #                 self.params = params_to_use
    #                 out = self.calculate(df)
    #             finally:
    #                 self.params = backup

    #         # assign same DataFrame reference to all mapped global param_set names
    #         for gname in info['global_ps_names']:
    #             results[asset_key][tf_key][gname] = out

    #     return results


    






