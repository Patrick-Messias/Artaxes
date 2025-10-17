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

    def _calculates_indicators(self): # Calculate all indicators for the provided data
        for ind in self.indicators.values():
            for asset in self.assets.values():
                df = asset.data_get(self.timeframe).copy()
                ind_values = ind.calculate_all_sets(df)

                self.data.setdefault(asset.name, {}).setdefault(self.timeframe, {}) \
                        .setdefault(ind.__class__.__name__, {})

                # Se for dict
                if isinstance(ind_values, dict):
                    for param_set, ind_obj in ind_values.items():
                        self.data[asset.name][self.timeframe][ind.__class__.__name__][param_set] = ind_obj
                # Se for lista
                elif isinstance(ind_values, list):
                    for idx, ind_obj in enumerate(ind_values):
                        self.data[asset.name][self.timeframe][ind.__class__.__name__][idx] = ind_obj

        return True

    def _generate_signals(self): # Generate entry signals based on indicators and assets
        """
        >0 = Long Entry Signal
        <0 = Short Entry Signal
        NaN = Neutral
        """
        self.signals = {}

        for asset_name, asset in self.assets.items():
            df = asset.data_get(self.timeframe).copy()
            signals_df = pd.DataFrame(index=df.index)

            # iterar pelas regras
            for rule_name, rule_func in self.entry_signal_rules.items():
                # cada função de regra recebe:
                # - df de preços
                # - dicionário com todas as séries de indicadores para este asset
                ind_series_dict = {}
                for ind_name, ind_param_dict in self.data[asset_name][self.timeframe].items():
                    # pegar o primeiro param_set de cada indicador
                    first_key = list(ind_param_dict.keys())[0]
                    ind_series_dict[ind_name.lower()] = ind_param_dict[first_key]

                # gera o sinal
                signals_df[rule_name] = rule_func(df, ind_series_dict)

            # combina sinais
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

        self.returns = {}  # Dict[asset_name, pd.DataFrame]

        for asset_name, asset in self.assets.items():
            df = asset.data_get(self.timeframe).copy()
            signals = self.signals[asset_name]

            # Cria um DataFrame que vai armazenar return por hold_period
            results = pd.DataFrame(index=self.foward_result_lookahead)
            results.index.name = 'lookahead'
            results['return'] = np.nan
            results['MAE'] = np.nan
            results['MFE'] = np.nan

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
                else:
                    returns_long = mae_long = mfe_long = pd.Series(dtype='float')

                # Short trades
                if mask_short.any():
                    returns_short = (price[mask_short] - future_prices[mask_short]) / price[mask_short]
                    mae_short = ((price[mask_short] - price[mask_short].rolling(hold_period, min_periods=1).max().shift(-hold_period+1)) / price[mask_short])
                    mfe_short = ((price[mask_short] - price[mask_short].rolling(hold_period, min_periods=1).min().shift(-hold_period+1)) / price[mask_short])
                else:
                    returns_short = mae_short = mfe_short = pd.Series(dtype='float')

                # Média dos retornos desse hold_period
                all_returns = pd.concat([returns_long, returns_short])
                all_mae = pd.concat([mae_long, mae_short])
                all_mfe = pd.concat([mfe_long, mfe_short])

                results.loc[hold_period, 'return'] = all_returns.sum()
                results.loc[hold_period, 'MAE'] = all_mae.mean()
                results.loc[hold_period, 'MFE'] = all_mfe.mean()

            self.returns[asset_name] = results

        return self.returns

    def analyse_results(self, x='lookahead', y='return', asset='combined', ma=5):
        if not hasattr(self, 'returns'):
            raise ValueError("Run _calculate_returns() first!")

        if asset == 'combined':
            # Plot combinado (como está agora)
            y_vals = pd.concat([df[y] for df in self.returns.values()], axis=1).mean(axis=1)
            
            plt.figure(figsize=(10, 5))
            plt.plot(y_vals.index, y_vals.values, marker='o', linestyle='-', color='b', alpha=0.5, label=f'{y.capitalize()}')
            if ma > 1:
                moving_avg = y_vals.rolling(window=ma, min_periods=1).mean()
                plt.plot(moving_avg.index, moving_avg.values, linestyle='-', color='red', linewidth=1, label=f'{ma}-Period MA')
            plt.xlabel('Forward Result Lookahead (Bars)')
            plt.ylabel(y.capitalize())
            plt.title(f'{y.capitalize()} vs Lookahead (Asset: {asset})')
            plt.grid(True, color='grey', alpha=0.2)
            plt.xticks(ticks=y_vals.index, labels=[int(x) for x in y_vals.index])
            plt.legend()
            plt.show()
            
        elif asset == 'all':
            # Plot individual para cada asset em subplots
            asset_names = list(self.returns.keys())
            n_assets = len(asset_names)

            if n_assets > 8:
                print("Too many assets to plot individually. Please select a specific asset or use 'combined' option.")
                return
            
            # Calcula layout automático dos subplots
            n_cols = 2  # Máximo de 2 colunas
            n_rows = (n_assets + n_cols - 1) // n_cols  # Arredonda para cima
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            #fig.suptitle(f'{y.capitalize()} vs Lookahead - All Assets', fontsize=16, y=0.95)
            
            # Se houver apenas 1 asset, axes não é array
            if n_assets == 1:
                axes = np.array([axes])
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, asset_name in enumerate(asset_names):
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col]
                
                y_vals = self.returns[asset_name][y]
                
                ax.plot(y_vals.index, y_vals.values, marker='o', linestyle='-', color='b', alpha=0.5, label=f'{y.capitalize()}')
                if ma > 1:
                    moving_avg = y_vals.rolling(window=ma, min_periods=1).mean()
                    ax.plot(moving_avg.index, moving_avg.values, linestyle='-', color='red', linewidth=1, label=f'{ma}-Period MA')
                
                if asset != 'all': 
                    ax.set_xlabel('Forward Result Lookahead (Bars)')
                    ax.set_ylabel(y.capitalize())
                ax.set_title(f'Asset: {asset_name}')
                ax.grid(True, color='grey', alpha=0.2)
                ax.set_xticks(y_vals.index)
                ax.set_xticklabels([int(x) for x in y_vals.index])
                ax.legend()
            
            # Remove subplots vazios se houver
            for i in range(n_assets, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                fig.delaxes(axes[row, col])
            
            plt.tight_layout()
            plt.show()
            
        else:
            # Plot para um asset específico
            y_vals = self.returns[asset][y]
            
            plt.figure(figsize=(10, 5))
            plt.plot(y_vals.index, y_vals.values, marker='o', linestyle='-', color='b', alpha=0.5, label=f'{y.capitalize()}')
            if ma > 1:
                moving_avg = y_vals.rolling(window=ma, min_periods=1).mean()
                plt.plot(moving_avg.index, moving_avg.values, linestyle='-', color='red', linewidth=1, label=f'{ma}-Period MA')
            plt.xlabel('Forward Result Lookahead (Bars)')
            plt.ylabel(y.capitalize())
            plt.title(f'{y.capitalize()} vs Lookahead (Asset: {asset})')
            plt.grid(True, color='grey', alpha=0.2)
            plt.xticks(ticks=y_vals.index, labels=[int(x) for x in y_vals.index])
            plt.legend()
            plt.show()

    def validate_predictive_power(self): # Validate the predictive power of the indicators and strategies used
        return True

    def run_monte_carlo_simulation(self): # Run Monte Carlo simulations to assess robustness of the findings
        return True
    
# ==================================================================================||

    def plot_param_heatmap(self, x='param_set', y='lookahead', z='return', asset='combined'):
        """
        Plota um heatmap 3D com a soma de resultados de todas as combinações de parâmetros
        e lookahead periods.
        """
        if not hasattr(self, 'signals'):
            raise ValueError("Run _generate_signals() first!")
        
        # Coletar dados de todos os parâmetros
        all_results = []
        
        # Para cada combinação de parâmetros dos indicadores
        for asset_name, asset_data in self.data.items():
            # Primeiro, coletar TODOS os parâmetros disponíveis para este asset
            all_indicators_params = {}
            for indicator_name, param_dict in asset_data[self.timeframe].items():
                all_indicators_params[indicator_name] = list(param_dict.keys())
            
            # Gerar todas as combinações possíveis de parâmetros
            from itertools import product
            param_combinations = list(product(*all_indicators_params.values()))
            
            for param_combo in param_combinations:
                # Criar dicionário com os parâmetros específicos para esta combinação
                specific_params = {}
                for i, indicator_name in enumerate(all_indicators_params.keys()):
                    param_set = param_combo[i]
                    specific_params[indicator_name] = {
                        param_set: asset_data[self.timeframe][indicator_name][param_set]
                    }
                
                # Gerar sinais para esta combinação específica de parâmetros
                signals = self._generate_signals_for_params(asset_name, specific_params)
                
                # Calcular returns para esta combinação
                returns_df = self._calculate_returns_for_signals(asset_name, signals, use_sum=(z=='return'))
                
                # Criar uma string única para identificar esta combinação de parâmetros
                param_set_str = "_".join([f"{ind}_{param}" for ind, param in zip(all_indicators_params.keys(), param_combo)])
                
                # Adicionar aos resultados
                for lookahead in returns_df.index:
                    all_results.append({
                        'asset': asset_name,
                        'indicator': 'combined',  # Indica que é combinação de múltiplos indicadores
                        'param_set': param_set_str,
                        'lookahead': lookahead,
                        'return': returns_df.loc[lookahead, 'return'],
                        'MAE': returns_df.loc[lookahead, 'MAE'],
                        'MFE': returns_df.loc[lookahead, 'MFE']
                    })
        
        # Resto do código permanece igual...
        # Criar DataFrame com todos os resultados
        results_df = pd.DataFrame(all_results)
        
        # Função para extrair números dos parâmetros para ordenação
        def extract_numeric_value(param_str):
            param_str = str(param_str)
            # Procura por números no parâmetro (incluindo decimais)
            import re
            numbers = re.findall(r'-?\d+\.?\d*', param_str)
            if numbers:
                return float(numbers[0])
            # Se não encontrar números, retorna o próprio string para ordenação alfabética
            return param_str
        
        # Ordenar os resultados
        if x == 'param_set':
            results_df['x_sort'] = results_df['param_set'].apply(extract_numeric_value)
            results_df = results_df.sort_values('x_sort')
        elif y == 'param_set':
            results_df['y_sort'] = results_df['param_set'].apply(extract_numeric_value)
            results_df = results_df.sort_values('y_sort')
        
        if asset == 'combined':
            # Agrupar por param_set e lookahead
            if z == 'return':
                # Para return, usar SOMA
                heatmap_data = results_df.groupby([x, y])[z].sum().unstack()
            else:
                # Para MAE/MFE, usar MÉDIA
                heatmap_data = results_df.groupby([x, y])[z].mean().unstack()
        else:
            # Filtrar por asset específico
            asset_data = results_df[results_df['asset'] == asset]
            if z == 'return':
                heatmap_data = asset_data.groupby([x, y])[z].sum().unstack()
            else:
                heatmap_data = asset_data.groupby([x, y])[z].mean().unstack()
        
        # Reordenar as linhas/colunas para garantir ordem numérica
        def sort_index_numeric(index):
            try:
                # Tentar ordenar numericamente
                sorted_items = sorted([(extract_numeric_value(str(item)), item) for item in index])
                return [item[1] for item in sorted_items]
            except:
                # Se falhar, usar ordenação alfabética
                return sorted(index)
        
        if x == 'param_set':
            heatmap_data = heatmap_data.reindex(sort_index_numeric(heatmap_data.index))
        if y == 'param_set':
            heatmap_data = heatmap_data.reindex(sort_index_numeric(heatmap_data.columns), axis=1)
        
        # Plotar heatmap
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111)
        
        # Criar heatmap com colormap apropriado
        if z == 'return':
            cmap = 'RdYlGn'  # Verde para positivo, Vermelho para negativo
            # Centralizar em 0 manualmente
            if not heatmap_data.empty:
                vmax = max(abs(heatmap_data.values.min()), abs(heatmap_data.values.max()))
                vmin = -vmax
            else:
                vmin, vmax = -1, 1
        else:
            cmap = 'viridis'  # Para MAE/MFE
            if not heatmap_data.empty:
                vmin = heatmap_data.values.min()
                vmax = heatmap_data.values.max()
            else:
                vmin, vmax = 0, 1
        
        if not heatmap_data.empty:
            im = ax.imshow(heatmap_data.values, cmap=cmap, aspect='auto', 
                        interpolation='nearest', vmin=vmin, vmax=vmax)
            
            # Configurar labels
            ax.set_xlabel(y.capitalize())
            ax.set_ylabel(x.capitalize())
            title_suffix = " (SUM)" if z == 'return' else " (MEAN)"
            ax.set_title(f'Heatmap {z.capitalize()} - {x} vs {y} (Asset: {asset}){title_suffix}')
            
            # Configurar ticks
            ax.set_xticks(range(len(heatmap_data.columns)))
            ax.set_xticklabels([str(x) for x in heatmap_data.columns], rotation=45)
            ax.set_yticks(range(len(heatmap_data.index)))
            ax.set_yticklabels([str(y) for y in heatmap_data.index])
            
            # Adicionar barra de cores
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(f'{z.capitalize()}' + (' (Sum)' if z == 'return' else ' (Mean)'))
            
            # Adicionar valores nas células (formatados baseado na métrica)
            for i in range(len(heatmap_data.index)):
                for j in range(len(heatmap_data.columns)):
                    value = heatmap_data.iloc[i, j]
                    if pd.isna(value):
                        continue
                        
                    if z == 'return':
                        text_val = f'{value:.1%}'  # Formato percentual para returns
                    else:
                        text_val = f'{value:.3f}'  # Formato decimal para MAE/MFE
                    
                    # Escolher cor do texto baseado no valor de fundo
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

    def _generate_signals_for_params(self, asset_name, specific_params):
        """
        Gera sinais para combinação específica de parâmetros
        """
        df = self.assets[asset_name].data_get(self.timeframe).copy()
        signals_df = pd.DataFrame(index=df.index)
        
        for rule_name, rule_func in self.entry_signal_rules.items():
            # Usar apenas os parâmetros específicos fornecidos
            ind_series_dict = {}
            for ind_name, ind_param_dict in specific_params.items():
                first_key = list(ind_param_dict.keys())[0]
                ind_series_dict[ind_name.lower()] = ind_param_dict[first_key]
            
            signals_df[rule_name] = rule_func(df, ind_series_dict)
        
        # Combina sinais
        combined_signal = pd.Series(0, index=df.index, dtype='float')
        if 'entry_long' in signals_df.columns:
            combined_signal[signals_df['entry_long']] = 1
        if 'entry_short' in signals_df.columns:
            combined_signal[signals_df['entry_short']] = -1
        
        combined_signal.replace(0, pd.NA, inplace=True)
        return combined_signal

    def _calculate_returns_for_signals(self, asset_name, signals, use_sum=False):
        """
        Calcula returns para sinais específicos
        use_sum: Se True, soma os retornos em vez de fazer média
        """
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
            
            # Long trades
            if mask_long.any():
                returns_long = (future_prices[mask_long] - price[mask_long]) / price[mask_long]
                mae_long = ((price[mask_long].rolling(hold_period, min_periods=1).min().shift(-hold_period+1) - price[mask_long]) / price[mask_long])
                mfe_long = ((price[mask_long].rolling(hold_period, min_periods=1).max().shift(-hold_period+1) - price[mask_long]) / price[mask_long])
            else:
                returns_long = mae_long = mfe_long = pd.Series(dtype='float')
            
            # Short trades
            if mask_short.any():
                returns_short = (price[mask_short] - future_prices[mask_short]) / price[mask_short]
                mae_short = ((price[mask_short] - price[mask_short].rolling(hold_period, min_periods=1).max().shift(-hold_period+1)) / price[mask_short])
                mfe_short = ((price[mask_short] - price[mask_short].rolling(hold_period, min_periods=1).min().shift(-hold_period+1)) / price[mask_short])
            else:
                returns_short = mae_short = mfe_short = pd.Series(dtype='float')
            
            # Concatenar todos os retornos
            all_returns = pd.concat([returns_long, returns_short])
            all_mae = pd.concat([mae_long, mae_short])
            all_mfe = pd.concat([mfe_long, mfe_short])
            
            # Usar soma ou média baseado no parâmetro
            if use_sum:
                results.loc[hold_period, 'return'] = all_returns.sum()
            else:
                results.loc[hold_period, 'return'] = all_returns.mean()
            
            results.loc[hold_period, 'MAE'] = all_mae.mean()
            results.loc[hold_period, 'MFE'] = all_mfe.mean()
        
        return results



def main() -> None:
    # Timeframe and Assets
    op_timeframe = 'M30'
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
    ma = MA(asset='CURR_ASSET', timeframe=op_timeframe, window=list(range(20, 200+1, 20)), type='sma', price_col='close')
    hurst = HURST(asset='CURR_ASSET', timeframe=op_timeframe, window=list(range(100, 200+1, 100)))
    kalman = KALMAN(asset='CURR_ASSET', timeframe=op_timeframe, R=[0.01, 0.1+0.01, 0.01], Q=[0.001, 0.01+0.001, 0.001])
    indicators={'ma': ma}#, 'kalman': kalman} #, 'hurst': hurst

    # Entry Rules L and S  
    def entry_long(df, ind_series_dict):
        ma = ind_series_dict['ma']
        #hurst = ind_series_dict['hurst']
        #kalman = ind_series_dict['kalman']
        return (df['close'] < ma) & (df['close'].shift(1) > ma.shift(1)) #& (df['close'] > kalman)
    
    def entry_short(df, ind_series_dict):
        ma = ind_series_dict['ma']
        #hurst = ind_series_dict['hurst']
        #kalman = ind_series_dict['kalman']
        return (df['close'] > ma) & (df['close'].shift(1) < ma.shift(1)) #& (df['close'] > kalman)
    
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
            foward_result_lookahead=range(1, 21+1),
            monte_carlo_simulations=1000
        )
    )

    operation._calculates_indicators()
    operation._generate_signals()
    returns = operation._calculate_returns()
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
    
    # Plots
    operation.analyse_results(x='foward_result_lookahead', y='return', asset='combined') #, asset='EURUSD'
    operation.plot_param_heatmap(x='param_set', y='lookahead', z='return', asset='combined')


if __name__ == "__main__":
    main()



