from dataclasses import dataclass, field
from BaseClass import BaseClass
from typing import Dict, Optional
from Indicator import Indicator
from Asset import Asset
import pandas as pd, matplotlib.pyplot as plt
import numpy as np
import uuid, sys

sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend\Indicators')
from MA import MA # type: ignore
from HURST import HURST # type: ignore
from KALMAN import KALMAN # type: ignore
from VIXFIX import VIXFIX # type: ignore
from VolModelManager import VolPredictor # type: ignore

"""
EdgeFinder is responsible for identifying trading edges based on indicators/strategies and assets
it does not execute trades or have any money management logic, only 1 timeframe, also doesn't use Model or Strat logic, only the signals
it's meant to be a simpler way to test out ideas and concepts before implementing them in a full Strat/Model context

for each asset:
   for each strategy:
      for each parameter combination:
- Relationship between residuals and predicted values
- Skew of distribuition of residuals
- Autocorrelation of residuals
- Heteroscedasticity of residuals
- R^2 of regression
"""

@dataclass
class EdgeFinderParams():
    name: str = field(default_factory=lambda: f'edgefinder_{uuid.uuid4()}')
    indicators: Dict[str, Indicator] = field(default_factory=dict) # Only uses signals from Strat, no money management or execution logic
    assets: Dict[str, Asset] = field(default_factory=dict)

    timeframe: str = 'D1'  # Timeframe for the analysis
    entry_signal_rules: Dict[str, callable] = field(default_factory=dict)  # Rules to generate entry signals
    exit_signal_rules: Optional[Dict[str, callable]] = field(default_factory=dict)   # Rules
    foward_result_lookahead: int = range(1, 63+1, 1)  # How many bars should we look forward to calculate MAE and MFE

    monte_carlo_simulations: int = 1000  # if 0 then no monte carlo simulation

class EdgeFinder(BaseClass): 
    def __init__(self, edgefinder_params: EdgeFinderParams):
        super().__init__()

        self.name = edgefinder_params.name
        self.indicators = edgefinder_params.indicators
        self.assets = edgefinder_params.assets

        self.timeframe = edgefinder_params.timeframe
        self.entry_signal_rules = edgefinder_params.entry_signal_rules
        self.exit_signal_rules = edgefinder_params.exit_signal_rules
        self.foward_result_lookahead = edgefinder_params.foward_result_lookahead

        self.monte_carlo_simulations = edgefinder_params.monte_carlo_simulations

        self.data = {} # Data from all assets concatenated for analysis
        self.all_param_results = {}  # Novo: Armazenar todos os resultados de parâmetros

    def _calculates_indicators(self):
        for ind_name_alias, ind in self.indicators.items():  # ind_name_alias: 'ma', 'mmm', 'volpredictor', etc.
            for asset in self.assets.values():
                df = asset.data_get(self.timeframe).copy()
                ind_values = ind.calculate_all_sets(df)

                # Cria chave única usando o alias
                self.data.setdefault(asset.name, {}).setdefault(self.timeframe, {}) \
                        .setdefault(ind_name_alias, {})

                if isinstance(ind_values, dict):
                    for param_set, ind_obj in ind_values.items():
                        self.data[asset.name][self.timeframe][ind_name_alias][param_set] = ind_obj
                elif isinstance(ind_values, list):
                    for idx, ind_obj in enumerate(ind_values):
                        self.data[asset.name][self.timeframe][ind_name_alias][idx] = ind_obj
        return True

    def _generate_signals(self):
        self.signals = {}
        for asset_name, asset in self.assets.items():
            df = asset.data_get(self.timeframe).copy()
            signals_df = pd.DataFrame(index=df.index)

            for rule_name, rule_func in self.entry_signal_rules.items():
                ind_series_dict = {}
                for ind_alias, ind_param_dict in self.data[asset_name][self.timeframe].items():
                    first_key = list(ind_param_dict.keys())[0]
                    ind_series_dict[ind_alias] = ind_param_dict[first_key]  # usa o alias correto
                signals_df[rule_name] = rule_func(df, ind_series_dict)

            combined_signal = pd.Series(0, index=df.index, dtype='float')
            if 'entry_long' in signals_df.columns:
                combined_signal[signals_df['entry_long']] = 1
            if 'entry_short' in signals_df.columns:
                combined_signal[signals_df['entry_short']] = -1
            combined_signal.replace(0, pd.NA, inplace=True)
            self.signals[asset_name] = combined_signal
        return self.signals


    def _calculate_returns(self):
            if not hasattr(self, 'signals'):
                raise ValueError("Signals not generated. Run _generate_signals() first.")

            self.returns = {}
            self.all_param_results = {}
            self.trades_data = {}  # Reset trades data

            for asset_name, asset in self.assets.items():
                df = asset.data_get(self.timeframe).copy()
                signals = self.signals[asset_name]

                results = pd.DataFrame(index=self.foward_result_lookahead)
                results.index.name = 'lookahead'
                results['return'] = np.nan # Total Sum Return
                results['avg_return'] = np.nan # Total Average Return # NOVO
                results['avg_return_long'] = np.nan  # Média de Retorno Long # NOVO
                results['avg_return_short'] = np.nan # Média de Retorno Short # NOVO
                results['MAE'] = np.nan
                results['MFE'] = np.nan

                # NOVO: Coletar trades individuais
                all_trades = []
                price = df['close']

                for hold_period in self.foward_result_lookahead:
                    future_prices = price.shift(-hold_period)
                    mask_long = signals == 1
                    mask_short = signals == -1

                    # Long trades
                    if mask_long.any():
                        returns_long = (future_prices[mask_long] - price[mask_long]) / price[mask_long]
                        mae_long = ((price[mask_long].rolling(hold_period, min_periods=1).min().shift(-hold_period+1) - price[mask_long]) / price[mask_long])
                        mfe_long = ((price[mask_long].rolling(hold_period, min_periods=1).max().shift(-hold_period+1) - price[mask_long]) / price[mask_long])
                        
                        # Coletar trades individuais
                        for ret in returns_long:
                            all_trades.append({'return': ret, 'type': 'long', 'hold_period': hold_period})
                    else:
                        returns_long = mae_long = mfe_long = pd.Series(dtype='float')

                    # Short trades
                    if mask_short.any():
                        returns_short = (price[mask_short] - future_prices[mask_short]) / price[mask_short]
                        mae_short = ((price[mask_short] - price[mask_short].rolling(hold_period, min_periods=1).max().shift(-hold_period+1)) / price[mask_short])
                        mfe_short = ((price[mask_short] - price[mask_short].rolling(hold_period, min_periods=1).min().shift(-hold_period+1)) / price[mask_short])
                        
                        # Coletar trades individuais
                        for ret in returns_short:
                            all_trades.append({'return': ret, 'type': 'short', 'hold_period': hold_period})
                    else:
                        returns_short = mae_short = mfe_short = pd.Series(dtype='float')

                    all_returns = pd.concat([returns_long, returns_short])
                    all_mae = pd.concat([mae_long, mae_short])
                    all_mfe = pd.concat([mfe_long, mfe_short])

                    results.loc[hold_period, 'return'] = all_returns.sum()
                    results.loc[hold_period, 'avg_return'] = all_returns.mean() # NOVO
                    results.loc[hold_period, 'avg_return_long'] = returns_long.mean() if not returns_long.empty else np.nan # NOVO
                    results.loc[hold_period, 'avg_return_short'] = returns_short.mean() if not returns_short.empty else np.nan # NOVO
                    results.loc[hold_period, 'MAE'] = all_mae.mean()
                    results.loc[hold_period, 'MFE'] = all_mfe.mean()

                self.returns[asset_name] = results
                self.trades_data[asset_name] = pd.DataFrame(all_trades)  # Armazenar trades

                # Calcular combinações de parâmetros
                asset_param_results = self._calculate_all_param_combinations(asset_name)
                self.all_param_results[asset_name] = asset_param_results

            return self.returns

    def _calculate_all_param_combinations(self, asset_name):
        asset_data = self.data[asset_name][self.timeframe]
        all_results = []
        
        all_indicators_params = {}
        for indicator_name, param_dict in asset_data.items():
            all_indicators_params[indicator_name] = list(param_dict.keys())
        
        from itertools import product
        param_combinations = list(product(*all_indicators_params.values()))
        
        for param_combo in param_combinations:
            specific_params = {}
            for i, indicator_name in enumerate(all_indicators_params.keys()):
                param_set = param_combo[i]
                specific_params[indicator_name] = {
                    param_set: asset_data[indicator_name][param_set]
                }
            
            signals = self._generate_signals_for_params(asset_name, specific_params)
            returns_df = self._calculate_returns_for_signals(asset_name, signals, use_sum=True)
            
            param_set_str = "_".join([f"{ind}_{param}" for ind, param in zip(all_indicators_params.keys(), param_combo)])
            
            for lookahead in returns_df.index:
                all_results.append({
                    'asset': asset_name,
                    'indicator': 'combined',
                    'param_set': param_set_str,
                    'lookahead': lookahead,
                    'return': returns_df.loc[lookahead, 'return'], # Apenas o retorno total agregado
                    'MAE': returns_df.loc[lookahead, 'MAE'],
                    'MFE': returns_df.loc[lookahead, 'MFE']
                })
        
        return all_results

    def _generate_signals_for_params(self, asset_name, specific_params):
        df = self.assets[asset_name].data_get(self.timeframe).copy()
        signals_df = pd.DataFrame(index=df.index)
        
        for rule_name, rule_func in self.entry_signal_rules.items():
            ind_series_dict = {}
            for ind_name, ind_param_dict in specific_params.items():
                first_key = list(ind_param_dict.keys())[0]
                ind_series_dict[ind_name.lower()] = ind_param_dict[first_key]
            signals_df[rule_name] = rule_func(df, ind_series_dict)
        
        combined_signal = pd.Series(0, index=df.index, dtype='float')
        if 'entry_long' in signals_df.columns:
            combined_signal[signals_df['entry_long']] = 1
        if 'entry_short' in signals_df.columns:
            combined_signal[signals_df['entry_short']] = -1
        combined_signal.replace(0, pd.NA, inplace=True)
        return combined_signal
        
    def _calculate_returns_for_signals(self, asset_name, signals, use_sum=False):
            df = self.assets[asset_name].data_get(self.timeframe).copy()
            price = df['close']
            
            results = pd.DataFrame(index=self.foward_result_lookahead)
            results.index.name = 'lookahead'
            results['return'] = np.nan
            results['MAE'] = np.nan
            results['MFE'] = np.nan
            
            for hold_period in self.foward_result_lookahead:
                future_prices = price.shift(-hold_period)
                mask_long = signals == 1
                mask_short = signals == -1
                
                if mask_long.any():
                    returns_long = (future_prices[mask_long] - price[mask_long]) / price[mask_long]
                    mae_long = ((price[mask_long].rolling(hold_period, min_periods=1).min().shift(-hold_period+1) - price[mask_long]) / price[mask_long])
                    mfe_long = ((price[mask_long].rolling(hold_period, min_periods=1).max().shift(-hold_period+1) - price[mask_long]) / price[mask_long])
                else:
                    returns_long = mae_long = mfe_long = pd.Series(dtype='float')
                
                if mask_short.any():
                    # **CÁLCULO DE RETORNO SHORT:** O lucro é quando o preço cai. entry_price - future_price.
                    # Se future_price < entry_price, o numerador é POSITIVO (Lucro).
                    returns_short = (price[mask_short] - future_prices[mask_short]) / price[mask_short]
                    mae_short = ((price[mask_short] - price[mask_short].rolling(hold_period, min_periods=1).max().shift(-hold_period+1)) / price[mask_short])
                    mfe_short = ((price[mask_short] - price[mask_short].rolling(hold_period, min_periods=1).min().shift(-hold_period+1)) / price[mask_short])
                else:
                    returns_short = mae_short = mfe_short = pd.Series(dtype='float')
                
                all_returns = pd.concat([returns_long, returns_short])
                all_mae = pd.concat([mae_long, mae_short])
                all_mfe = pd.concat([mfe_long, mfe_short])
                
                # Cálculo do retorno: soma total para o heatmap/best-parameters
                if use_sum:
                    results.loc[hold_period, 'return'] = all_returns.sum()
                else:
                    results.loc[hold_period, 'return'] = all_returns.mean()
                
                results.loc[hold_period, 'MAE'] = all_mae.mean()
                results.loc[hold_period, 'MFE'] = all_mfe.mean()
            
            return results
   
    def analyse_results(self, x='lookahead', y='return', asset='combined', ma=5):
        if not hasattr(self, 'returns'):
            raise ValueError("Run _calculate_returns() first!")

        plot_separate_long_short = y in ['return', 'avg_return']

        # Função para calcular média positiva e negativa limitada ao lookahead
        def get_pos_neg_means(y_vals):
            pos_mean = y_vals[y_vals > 0].mean() if not y_vals[y_vals > 0].empty else 0
            neg_mean = y_vals[y_vals < 0].mean() if not y_vals[y_vals < 0].empty else 0
            return pos_mean, neg_mean

        if asset == 'combined':
            if plot_separate_long_short:
                y_vals_long = pd.concat([df['avg_return_long'] for df in self.returns.values()], axis=1).mean(axis=1)
                y_vals_short = pd.concat([df['avg_return_short'] for df in self.returns.values()], axis=1).mean(axis=1)

                # Limita ao período de lookahead
                lookahead_periods = y_vals_long.index
                y_vals_long = y_vals_long.loc[lookahead_periods]
                y_vals_short = y_vals_short.loc[lookahead_periods]

                # Calcula médias positiva e negativa
                pos_mean, neg_mean = get_pos_neg_means(pd.concat([y_vals_long, y_vals_short]))

                plt.figure(figsize=(12, 6))
                # Linhas médias positivas e negativas
                plt.axhline(pos_mean, color='darkgreen', linestyle='-', linewidth=0.8, alpha=0.5, label='Mean Positive')
                plt.axhline(neg_mean, color='darkred', linestyle='-', linewidth=0.8, alpha=0.5, label='Mean Negative')

                # Plot Long
                plt.plot(y_vals_long.index, y_vals_long.values, marker='o', linestyle='-', color='darkgreen', alpha=0.8, label='Avg Return Long')
                if ma > 1:
                    plt.plot(y_vals_long.rolling(ma, min_periods=1).mean(), linestyle='-', color='g', alpha=0.8, linewidth=0.5, label=f'Long {ma}-Period MA')
                # Plot Short
                plt.plot(y_vals_short.index, y_vals_short.values, marker='o', linestyle='-', color='darkred', alpha=0.8, label='Avg Return Short')
                if ma > 1:
                    plt.plot(y_vals_short.rolling(ma, min_periods=1).mean(), linestyle='-', color='r', alpha=0.8, linewidth=0.5, label=f'Short {ma}-Period MA')

                plt.xlabel('Forward Result Lookahead (Bars)')
                plt.ylabel('Average Return (%)')
                plt.title(f'Average Return vs Lookahead - Long vs Short (Asset: {asset})')
                plt.grid(True, color='grey', alpha=0.2)
                plt.xticks(ticks=y_vals_long.index, labels=[int(x) for x in y_vals_long.index])
                plt.legend()
                plt.show()

            else:
                y_col = 'avg_return' if y == 'return' else y
                y_vals = pd.concat([df[y_col] for df in self.returns.values()], axis=1).mean(axis=1)
                lookahead_periods = y_vals.index
                y_vals = y_vals.loc[lookahead_periods]

                pos_mean, neg_mean = get_pos_neg_means(y_vals)

                plt.figure(figsize=(10, 5))
                # Linhas médias positivas e negativas
                plt.axhline(pos_mean, color='darkgreen', linestyle='-', linewidth=0.8, alpha=0.5, label='Mean Positive')
                plt.axhline(neg_mean, color='darkred', linestyle='-', linewidth=0.8, alpha=0.5, label='Mean Negative')
                # Plot
                plt.plot(y_vals.index, y_vals.values, marker='o', linestyle='-', color='b', alpha=0.8, label=f'{y.capitalize()}')
                if ma > 1:
                    plt.plot(y_vals.rolling(ma, min_periods=1).mean(), linestyle='-', color='r', alpha=0.8, linewidth=0.5, label=f'{ma}-Period MA')

                plt.xlabel('Forward Result Lookahead (Bars)')
                plt.ylabel(y.capitalize() + ' (%)')
                plt.title(f'{y.capitalize()} vs Lookahead (Asset: {asset})')
                plt.grid(True, color='grey', alpha=0.2)
                plt.xticks(ticks=y_vals.index, labels=[int(x) for x in y_vals.index])
                plt.legend()
                plt.show()

        else:
            # === Asset específico ===
            if plot_separate_long_short:
                y_vals_long = self.returns[asset]['avg_return_long']
                y_vals_short = self.returns[asset]['avg_return_short']
                lookahead_periods = y_vals_long.index
                y_vals_long = y_vals_long.loc[lookahead_periods]
                y_vals_short = y_vals_short.loc[lookahead_periods]

                pos_mean, neg_mean = get_pos_neg_means(pd.concat([y_vals_long, y_vals_short]))

                plt.figure(figsize=(12, 6))
                plt.axhline(pos_mean, color='darkgreen', linestyle='-', linewidth=0.5, alpha=0.5, label='Mean Positive')
                plt.axhline(neg_mean, color='darkred', linestyle='-', linewidth=0.5, alpha=0.5, label='Mean Negative')

                plt.plot(y_vals_long.index, y_vals_long.values, marker='o', linestyle='-', color='darkgreen', alpha=0.8, label='Avg Return Long')
                plt.plot(y_vals_short.index, y_vals_short.values, marker='o', linestyle='-', color='darkred', alpha=0.8, label='Avg Return Short')
                if ma > 1:
                    plt.plot(y_vals_long.rolling(ma, min_periods=1).mean(), linestyle='-', color='g', alpha=0.8, linewidth=0.75, label=f'Long {ma}-Period MA')
                    plt.plot(y_vals_short.rolling(ma, min_periods=1).mean(), linestyle='-', color='r', alpha=0.8, linewidth=0.75, label=f'Short {ma}-Period MA')

                plt.xlabel('Forward Result Lookahead (Bars)')
                plt.ylabel('Average Return (%)')
                plt.title(f'Average Return vs Lookahead - Long vs Short (Asset: {asset})')
                plt.grid(True, color='grey', alpha=0.2)
                plt.xticks(ticks=y_vals_long.index, labels=[int(x) for x in y_vals_long.index])
                plt.legend()
                plt.show()

            else:
                y_col = 'avg_return' if y == 'return' else y
                y_vals = self.returns[asset][y_col]
                lookahead_periods = y_vals.index
                y_vals = y_vals.loc[lookahead_periods]

                pos_mean, neg_mean = get_pos_neg_means(y_vals)

                plt.figure(figsize=(10, 5))
                plt.axhline(pos_mean, color='darkgreen', linestyle='-', linewidth=0.8, alpha=0.5, label='Mean Positive')
                plt.axhline(neg_mean, color='darkred', linestyle='-', linewidth=0.8, alpha=0.5, label='Mean Negative')
                plt.plot(y_vals.index, y_vals.values, marker='o', linestyle='-', color='b', alpha=1, label=f'{y.capitalize()}')
                if ma > 1:
                    plt.plot(y_vals.rolling(ma, min_periods=1).mean(), linestyle='-', color='red', linewidth=1, label=f'{ma}-Period MA')

                plt.xlabel('Forward Result Lookahead (Bars)')
                plt.ylabel(y.capitalize() + ' (%)')
                plt.title(f'{y.capitalize()} vs Lookahead (Asset: {asset})')
                plt.grid(True, color='grey', alpha=0.2)
                plt.xticks(ticks=y_vals.index, labels=[int(x) for x in y_vals.index])
                plt.legend()
                plt.show()


    def plot_param_heatmap(self, x='param_set', y='lookahead', z='return', asset='combined'):
        """
        Plota um heatmap 3D usando os dados já calculados
        """
        if not hasattr(self, 'all_param_results'):
            raise ValueError("Run _calculate_returns() first to generate parameter results!")
        
        # Usar os dados já calculados
        all_results = []
        for asset_name, asset_results in self.all_param_results.items():
            all_results.extend(asset_results)
        
        # Resto do código permanece igual (processamento e plotagem)
        results_df = pd.DataFrame(all_results)
        
        # Função para extrair números dos parâmetros para ordenação
        def extract_numeric_value(param_str):
            param_str = str(param_str)
            import re
            numbers = re.findall(r'-?\d+\.?\d*', param_str)
            if numbers:
                return float(numbers[0])
            return param_str
        
        # Ordenar os resultados
        if x == 'param_set':
            results_df['x_sort'] = results_df['param_set'].apply(extract_numeric_value)
            results_df = results_df.sort_values('x_sort')
        elif y == 'param_set':
            results_df['y_sort'] = results_df['param_set'].apply(extract_numeric_value)
            results_df = results_df.sort_values('y_sort')
        
        if asset == 'combined':
            if z == 'return':
                heatmap_data = results_df.groupby([x, y])[z].sum().unstack()
            else:
                heatmap_data = results_df.groupby([x, y])[z].mean().unstack()
        else:
            asset_data = results_df[results_df['asset'] == asset]
            if z == 'return':
                heatmap_data = asset_data.groupby([x, y])[z].sum().unstack()
            else:
                heatmap_data = asset_data.groupby([x, y])[z].mean().unstack()
        
        # Reordenar as linhas/colunas para garantir ordem numérica
        def sort_index_numeric(index):
            try:
                sorted_items = sorted([(extract_numeric_value(str(item)), item) for item in index])
                return [item[1] for item in sorted_items]
            except:
                return sorted(index)
        
        if x == 'param_set':
            heatmap_data = heatmap_data.reindex(sort_index_numeric(heatmap_data.index))
        if y == 'param_set':
            heatmap_data = heatmap_data.reindex(sort_index_numeric(heatmap_data.columns), axis=1)
        
        # Plotar heatmap
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111)
        
        if z == 'return':
            cmap = 'RdYlGn'
            if not heatmap_data.empty:
                vmax = max(abs(heatmap_data.values.min()), abs(heatmap_data.values.max()))
                vmin = -vmax
            else:
                vmin, vmax = -1, 1
        else:
            cmap = 'viridis'
            if not heatmap_data.empty:
                vmin = heatmap_data.values.min()
                vmax = heatmap_data.values.max()
            else:
                vmin, vmax = 0, 1
        
        if not heatmap_data.empty:
            im = ax.imshow(heatmap_data.values, cmap=cmap, aspect='auto', 
                           interpolation='nearest', vmin=vmin, vmax=vmax)
            
            ax.set_xlabel(y.capitalize())
            ax.set_ylabel(x.capitalize())
            title_suffix = " (SUM)" if z == 'return' else " (MEAN)"
            ax.set_title(f'Heatmap {z.capitalize()} - {x} vs {y} (Asset: {asset}){title_suffix}')
            
            ax.set_xticks(range(len(heatmap_data.columns)))
            ax.set_xticklabels([str(x) for x in heatmap_data.columns], rotation=45)
            ax.set_yticks(range(len(heatmap_data.index)))
            ax.set_yticklabels([str(y) for y in heatmap_data.index])
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(f'{z.capitalize()}' + (' (Sum)' if z == 'return' else ' (Mean)'))
            
            for i in range(len(heatmap_data.index)):
                for j in range(len(heatmap_data.columns)):
                    value = heatmap_data.iloc[i, j]
                    if pd.isna(value):
                        continue
                        
                    if z == 'return':
                        text_val = f'{value:.1%}'
                    else:
                        text_val = f'{value:.3f}'
                    
                    normalized_value = (value - vmin) / (vmax - vmin) if vmax != vmin else 0.5
                    text_color = 'white' if normalized_value < 0.5 else 'black'
                    
                    text = ax.text(j, i, text_val,
                                   ha="center", va="center", color=text_color, 
                                   fontsize=8, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'No data for Heatmap {z.capitalize()} - {x} vs {y} (Asset: {asset})')
        
        plt.tight_layout()
        plt.show()
        
        return heatmap_data
    
    def print_best_parameters(self, metric='return', top_n=5):
        """Imprime os melhores parâmetros encontrados"""
        if not hasattr(self, 'all_param_results'):
            print("Run _calculate_returns() first!")
            return
        
        all_results = []
        for asset_name, asset_results in self.all_param_results.items():
            for result in asset_results:
                all_results.append(result)
        
        results_df = pd.DataFrame(all_results)
        
        # Encontrar melhores combinações
        best_combinations = results_df.groupby('param_set')[metric].mean().nlargest(top_n)
        
        print(f"\n-> Best {top_n} Parameter Combinations ({metric.upper()})")
        for i, (param_set, value) in enumerate(best_combinations.items(), 1):
            if metric == 'return':
                print(f"{i}. {value:.2%} >>> {param_set}")
            else:
                print(f"{i}. {value:.4f} >>> {param_set}")
        
        return best_combinations

    #======================================================================================================================|| Metrics

    def validate_predictive_power(self): # Validate the predictive power of the indicators and strategies used
        return True

    def run_monte_carlo_simulation(self): # Run Monte Carlo simulations to assess robustness of the findings
        return True

def main() -> None:
    # Timeframe and Assets
    op_timeframe = 'H1'
    eurusd = Asset(
        name='EURUSD',
        type='currency_pair',
        market='forex',
        data_path=f'C:\\Users\\Patrick\\Desktop\\Artaxes Portfolio\\MAIN\\MT5_Dados\\Forex',
        timeframe=[op_timeframe])
    gbpusd = Asset(
        name='GBPUSD',
        type='currency_pair',
        market='forex',
        data_path=f'C:\\Users\\Patrick\\Desktop\\Artaxes Portfolio\\MAIN\\MT5_Dados\\Forex',
        timeframe=[op_timeframe])
    usdjpy = Asset(
        name='USDJPY',
        type='currency_pair',
        market='forex',
        data_path=f'C:\\Users\\Patrick\\Desktop\\Artaxes Portfolio\\MAIN\\MT5_Dados\\Forex',
        timeframe=[op_timeframe])
    audcad = Asset(
        name='AUDCAD',
        type='currency_pair',
        market='forex',
        data_path=f'C:\\Users\\Patrick\\Desktop\\Artaxes Portfolio\\MAIN\\MT5_Dados\\Forex',
        timeframe=[op_timeframe])
    nzdusd = Asset(
        name='NZDUSD',
        type='currency_pair',
        market='forex',
        data_path=f'C:\\Users\\Patrick\\Desktop\\Artaxes Portfolio\\MAIN\\MT5_Dados\\Forex',
        timeframe=[op_timeframe])
    usdchf = Asset(
        name='USDCHF',
        type='currency_pair',
        market='forex',
        data_path=f'C:\\Users\\Patrick\\Desktop\\Artaxes Portfolio\\MAIN\\MT5_Dados\\Forex',
        timeframe=[op_timeframe])
    #assets = {eurusd.name: eurusd, gbpusd.name: gbpusd, usdjpy.name: usdjpy, audcad.name: audcad, nzdusd.name: nzdusd, usdchf.name: usdchf}
    assets = {eurusd.name: eurusd}

    # Indicators
    ma = MA(asset='CURR_ASSET', timeframe=op_timeframe, window=list(range(7, 7+1, 7)), type='sma', price_col='close')
    mmm = MA(asset='CURR_ASSET', timeframe=op_timeframe, window=list(range(21, 21+1, 21)), type='sma', price_col='close')
    hurst = HURST(asset='CURR_ASSET', timeframe=op_timeframe, window=list(range(63, 63+1, 100)))
    kalman = KALMAN(asset='CURR_ASSET', timeframe=op_timeframe, R=[0.01, 0.1+0.01, 0.01], Q=[0.001, 0.01+0.001, 0.001])
    vixfix = VIXFIX(asset='CURR_ASSET', timeframe=op_timeframe, window=list(range(22, 22+1, 2)))
    volpredictor = VolPredictor(asset='CURR_ASSET', timeframe=op_timeframe)
    indicators={'volpredictor': volpredictor, 'ma': ma, 'mmm': mmm, 'hurst': hurst} #{#, 'kalman': kalman} #, 'hurst': hurst

    # Entry Rules L and S  
    def entry_long(df, ind_series_dict):
        volpredictor = ind_series_dict['volpredictor']
        ma = ind_series_dict['ma']
        mmm = ind_series_dict['mmm']
        hurst = ind_series_dict['hurst']

        signal = (volpredictor > 0) & (ma < mmm) #& (hurst < 0.5) #& (df['close'].shift(1) < df['open'].shift(1)) #& (df['close'].shift(1) < df['open'].shift(1)) #
        signal = signal.fillna(False)
        return signal

    def entry_short(df, ind_series_dict):
        volpredictor = ind_series_dict['volpredictor']
        ma = ind_series_dict['ma']
        mmm = ind_series_dict['mmm']
        hurst = ind_series_dict['hurst']

        signal = (volpredictor > 0) & (ma > mmm) #& (hurst < 0.5) #& (df['close'].shift(1) > df['open'].shift(1)) #
        signal = signal.fillna(False)
        return signal

    entry_signal_rules = {'entry_long': entry_long, 'entry_short': entry_short}

    # Edge Finder Operation
    operation = EdgeFinder(
        EdgeFinderParams(
            name='SMA_Cross',
            indicators=indicators,
            assets=assets,
            timeframe=op_timeframe,
            entry_signal_rules=entry_signal_rules,
            exit_signal_rules=None,
            foward_result_lookahead=range(1, 5+1),
            monte_carlo_simulations=1000
        )
    )

    # Execution
    operation._calculates_indicators()
    operation._generate_signals()
    returns = operation._calculate_returns() 
    
    # Metrics
    #operation.print_detailed_metrics(asset='combined')
    print("===============|| Edge Finder Results (Mean) ||===============")
    pnl_sum, mfe_sum, mae_sum = 0, 0, 0
    for asset_name, ret in returns.items():
        pnl = ret['return'].dropna().mean()
        mae = ret['MAE'].mean()
        mfe = ret['MFE'].mean()
        pnl_sum += pnl
        mae_sum += mae
        mfe_sum += mfe
        print(f"{asset_name} >>> Returns: {pnl*100:.2f}%, MAE: {mae*100:.3f}%, MFE: {mfe*100:.3f}%")
    num_assets = int(len(returns))
    print(f"COMBINED >>> Returns: {(pnl_sum/num_assets)*100:.2f}%, MAE: {(mae_sum/num_assets)*100:.3f}%, MFE: {(mfe_sum/num_assets)*100:.3f}%")

    operation.print_best_parameters(metric='return', top_n=5)
    
    # Plots
    operation.analyse_results(x='foward_result_lookahead', y='return', asset='combined') #, asset='EURUSD'
    #operation.plot_param_heatmap(x='param_set', y='lookahead', z='return', asset='combined')



if __name__ == "__main__":
    main()


"""

    
    def _calculate_global_trade_metrics(self):
        # Calcula métricas globais de TODOS os trades de TODOS os assets
        if not hasattr(self, 'trades_data'):
            return {}
        
        all_trades = []
        for asset_name, trades_df in self.trades_data.items():
            if len(trades_df) > 0:
                all_trades.append(trades_df)
        
        if not all_trades:
            return {}
        
        # Combinar todos os trades de todos os assets
        global_trades = pd.concat(all_trades, ignore_index=True)
        
        if len(global_trades) == 0:
            return {}
        
        win_rate = (global_trades['return'] > 0).sum() / len(global_trades) * 100
        avg_win = global_trades[global_trades['return'] > 0]['return'].mean()
        avg_loss = global_trades[global_trades['return'] <= 0]['return'].mean()
        
        # Calcular EV global
        win_rate_pct = win_rate / 100
        ev = (win_rate_pct * (avg_win if not pd.isna(avg_win) else 0)) + \
            ((1 - win_rate_pct) * (avg_loss if not pd.isna(avg_loss) else 0))
        
        winning_trades = global_trades[global_trades['return'] > 0]
        losing_trades = global_trades[global_trades['return'] <= 0]
        
        return {
            'total_trades': len(global_trades),
            'win_rate': win_rate,
            'expected_value': ev,
            'avg_win': avg_win if not pd.isna(avg_win) else 0,
            'avg_loss': avg_loss if not pd.isna(avg_loss) else 0,
            'largest_win': winning_trades['return'].max() if len(winning_trades) > 0 else 0,
            'largest_loss': losing_trades['return'].min() if len(losing_trades) > 0 else 0,
            'profit_factor': abs(winning_trades['return'].sum() / losing_trades['return'].sum()) if losing_trades['return'].sum() != 0 else float('inf')
        }

    def print_detailed_metrics(self, asset='combined'):
        # Imprime métricas detalhadas incluindo EV, win rate real, etc.
        if not hasattr(self, 'returns') or not hasattr(self, 'signals'):
            raise ValueError("Run _calculate_returns() and _generate_signals() first!")
        
        print("=" * 80)
        print("EDGE FINDER - DETAILED METRICS REPORT")
        print("=" * 80)
        
        if asset == 'combined':
            self._print_combined_metrics()
        else:
            self._print_asset_metrics(asset)
        
        # Melhores parâmetros
        #if hasattr(self, 'all_param_results'):
        #    self.print_best_parameters(metric='return', top_n=5)

    def _print_combined_metrics(self):
        # Imprime métricas combinadas para todos os assets
        returns = self.returns
        signals = self.signals
        
        num_assets = len(returns)
        pnl_sum, mfe_sum, mae_sum = 0, 0, 0
        total_signals = 0
        total_long_signals = 0
        total_short_signals = 0
        
        # NOVO: Métricas agregadas de trades reais
        combined_trade_metrics = {
            'total_trades': 0,
            'win_rate_sum': 0,
            'expected_value_sum': 0,
            'avg_win_sum': 0,
            'avg_loss_sum': 0,
            'profit_factor_sum': 0,
            'assets_with_trades': 0
        }
        
        print(f"COMBINED RESULTS ({num_assets} assets)")
        print("-" * 80)
        
        for asset_name, ret_df in returns.items():
            pnl = ret_df['return'].dropna().mean()
            mae = ret_df['MAE'].mean()
            mfe = ret_df['MFE'].mean()
            pnl_sum += pnl
            mae_sum += mae
            mfe_sum += mfe
            
            asset_signals = signals[asset_name]
            asset_total_signals = asset_signals.count()
            asset_long_signals = (asset_signals == 1).sum()
            asset_short_signals = (asset_signals == -1).sum()
            
            total_signals += asset_total_signals
            total_long_signals += asset_long_signals
            total_short_signals += asset_short_signals
            
            # Métricas de trades reais
            trade_metrics = self._calculate_trade_metrics(asset_name)
            
            print(f"{asset_name:10} | Ret: {pnl*100:6.2f}% | MAE: {mae*100:6.3f}% | MFE: {mfe*100:6.3f}% | "
                f"Trades: {trade_metrics['total_trades']:4d} | Win: {trade_metrics['win_rate']:5.1f}% | "
                f"EV: {trade_metrics['expected_value']*100:6.3f}% | PF: {trade_metrics['profit_factor']:5.2f}")
            
            # NOVO: Acumular métricas para cálculo da média combinada
            if trade_metrics['total_trades'] > 0:
                combined_trade_metrics['total_trades'] += trade_metrics['total_trades']
                combined_trade_metrics['win_rate_sum'] += trade_metrics['win_rate']
                combined_trade_metrics['expected_value_sum'] += trade_metrics['expected_value']
                combined_trade_metrics['avg_win_sum'] += trade_metrics['avg_win']
                combined_trade_metrics['avg_loss_sum'] += trade_metrics['avg_loss']
                if trade_metrics['profit_factor'] != float('inf'):
                    combined_trade_metrics['profit_factor_sum'] += trade_metrics['profit_factor']
                combined_trade_metrics['assets_with_trades'] += 1
        
        if num_assets > 0:
            avg_pnl = pnl_sum / num_assets
            avg_mae = mae_sum / num_assets
            avg_mfe = mfe_sum / num_assets
            
            long_ratio = (total_long_signals / total_signals * 100) if total_signals > 0 else 0
            short_ratio = (total_short_signals / total_signals * 100) if total_signals > 0 else 0
            
            print("-" * 80)
            print(f"{'COMBINED':10} | Ret: {avg_pnl*100:6.2f}% | MAE: {avg_mae*100:6.3f}% | MFE: {avg_mfe*100:6.3f}% | "
                f"{'':9} | {'':9} | {'':11} | Long/Short: {long_ratio:.1f}%/{short_ratio:.1f}%")
            
            # NOVO: Mostrar métricas combinadas de trades reais
            if combined_trade_metrics['assets_with_trades'] > 0:
                print(f"\nCOMBINED REAL TRADE METRICS (Average of {combined_trade_metrics['assets_with_trades']} assets):")
                print(f"  Total Trades:     {combined_trade_metrics['total_trades']:8d}")
                print(f"  Avg Win Rate:     {combined_trade_metrics['win_rate_sum'] / combined_trade_metrics['assets_with_trades']:8.1f}%")
                print(f"  Avg Expected Value: {((combined_trade_metrics['expected_value_sum'] / combined_trade_metrics['assets_with_trades'])*100):8.3f}%")
                print(f"  Avg Win:          {((combined_trade_metrics['avg_win_sum'] / combined_trade_metrics['assets_with_trades'])*100):8.3f}%")
                print(f"  Avg Loss:         {((combined_trade_metrics['avg_loss_sum'] / combined_trade_metrics['assets_with_trades'])*100):8.3f}%")
                if combined_trade_metrics['profit_factor_sum'] > 0:
                    print(f"  Avg Profit Factor: {combined_trade_metrics['profit_factor_sum'] / combined_trade_metrics['assets_with_trades']:8.2f}")

    def _print_asset_metrics(self, asset_name):
        # Imprime métricas detalhadas para um asset específico
        if asset_name not in self.returns:
            print(f"Asset {asset_name} not found!")
            return
        
        ret_df = self.returns[asset_name]
        signals = self.signals[asset_name]
        
        # Métricas básicas
        pnl = ret_df['return'].dropna().mean()
        mae = ret_df['MAE'].mean()
        mfe = ret_df['MFE'].mean()
        
        # Métricas de sinais
        total_signals = signals.count()
        long_signals = (signals == 1).sum()
        short_signals = (signals == -1).sum()
        long_ratio = (long_signals / total_signals * 100) if total_signals > 0 else 0
        short_ratio = (short_signals / total_signals * 100) if total_signals > 0 else 0
        
        # Métricas de trades reais
        trade_metrics = self._calculate_trade_metrics(asset_name)
        
        print(f"\nASSET: {asset_name}")
        print("=" * 50)
        print(f"STRATEGY METRICS:")
        print(f"  Return:           {pnl*100:8.2f}%")
        print(f"  MAE:              {mae*100:8.3f}%")
        print(f"  MFE:              {mfe*100:8.3f}%")
        print(f"  Total Signals:    {total_signals:8d}")
        print(f"  Long/Short:       {long_ratio:5.1f}% / {short_ratio:5.1f}%")
        
        if trade_metrics:
            print(f"\nREAL TRADE METRICS:")
            print(f"  Total Trades:     {trade_metrics['total_trades']:8d}")
            print(f"  Win Rate:         {trade_metrics['win_rate']:8.1f}%")
            print(f"  Expected Value:   {trade_metrics['expected_value']*100:8.3f}%")
            print(f"  Avg Win:          {trade_metrics['avg_win']*100:8.3f}%")
            print(f"  Avg Loss:         {trade_metrics['avg_loss']*100:8.3f}%")
            print(f"  Profit Factor:    {trade_metrics['profit_factor']:8.2f}")
            print(f"  Largest Win:      {trade_metrics['largest_win']*100:8.3f}%")
            print(f"  Largest Loss:     {trade_metrics['largest_loss']*100:8.3f}%")
            
            # Risk-Reward
            if trade_metrics['avg_loss'] != 0:
                risk_reward = abs(trade_metrics['avg_win'] / trade_metrics['avg_loss'])
                print(f"  Win/Loss Ratio:   {risk_reward:8.2f}")

    def _calculate_real_win_rate(self, asset_name):
        # Calcula win rate baseado em trades individuais
        if not hasattr(self, 'trades_data') or asset_name not in self.trades_data:
            return 0
        
        trades = self.trades_data[asset_name]
        if len(trades) == 0:
            return 0
        
        winning_trades = (trades['return'] > 0).sum()
        return (winning_trades / len(trades)) * 100

    def _calculate_ev(self, asset_name):
        # Calcula Expected Value
        if not hasattr(self, 'trades_data') or asset_name not in self.trades_data:
            return 0
        
        trades = self.trades_data[asset_name]
        if len(trades) == 0:
            return 0
        
        win_rate = self._calculate_real_win_rate(asset_name) / 100
        avg_win = trades[trades['return'] > 0]['return'].mean()
        avg_loss = trades[trades['return'] <= 0]['return'].mean()
        
        ev = (win_rate * (avg_win if not pd.isna(avg_win) else 0)) + \
            ((1 - win_rate) * (avg_loss if not pd.isna(avg_loss) else 0))
        
        return ev

    def _calculate_trade_metrics(self, asset_name):
        # Retorna métricas completas de trades
        if not hasattr(self, 'trades_data') or asset_name not in self.trades_data:
            return {}
        
        trades = self.trades_data[asset_name]
        if len(trades) == 0:
            return {}
        
        win_rate = self._calculate_real_win_rate(asset_name)
        ev = self._calculate_ev(asset_name)
        
        winning_trades = trades[trades['return'] > 0]
        losing_trades = trades[trades['return'] <= 0]
        
        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'expected_value': ev,
            'avg_win': winning_trades['return'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['return'].mean() if len(losing_trades) > 0 else 0,
            'largest_win': winning_trades['return'].max() if len(winning_trades) > 0 else 0,
            'largest_loss': losing_trades['return'].min() if len(losing_trades) > 0 else 0,
            'profit_factor': abs(winning_trades['return'].sum() / losing_trades['return'].sum()) if losing_trades['return'].sum() != 0 else float('inf')
        }



OLD

    # num_assets = int(len(returns))
    # pnl_sum, mfe_sum, mae_sum = 0, 0, 0
    # for asset_name, ret in returns.items():
    #     pnl = ret['return'].dropna().mean()
    #     mae = ret['MAE'].mean()
    #     mfe = ret['MFE'].mean()
    #     pnl_sum += pnl
    #     mae_sum += mae
    #     mfe_sum += mfe
    #     print(f"{asset_name} >>> Returns: {pnl*100:.2f}%, MAE: {mae*100:.3f}%, MFE: {mfe*100:.3f}%")
    # if num_assets>1: print(f"COMBINED >>> Returns: {(pnl_sum/num_assets)*100:.2f}%, MAE: {(mae_sum/num_assets)*100:.3f}%, MFE: {(mfe_sum/num_assets)*100:.3f}%")



"""


