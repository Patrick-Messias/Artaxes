from dataclasses import dataclass, field
from BaseClass import BaseClass
from typing import Dict, Optional
from Indicator import Indicator
from Asset import Asset
import pandas as pd
import uuid, sys

sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend\Indicators')
from MA import MA # type: ignore

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
    indicator: Dict[str, Indicator] = field(default_factory=dict) # Only uses signals from Strat, no money management or execution logic
    assets: Dict[str, Asset] = field(default_factory=dict)

    timeframe: str = 'D1'  # Timeframe for the analysis
    entry_signal_rules: Dict[str, callable] = field(default_factory=dict)  # Rules to generate entry signals
    exit_signal_rules: Optional[Dict[str, callable]] = field(default_factory=dict)   # Rules
    foward_result_lookback: int = range(1, 63+1, 1)  # How many bars should we look forward to calculate MAE and MFE

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
        self.foward_result_lookback = edgefinder_params.foward_result_lookback

        self.monte_carlo_simulations = edgefinder_params.monte_carlo_simulations

        self.data = Dict[str, pd.DataFrame] # Data from all assets concatenated for analysis

    def _calculates_indicators(self): # Calculate all indicators for the provided data
        for ind in self.indicators.values():
            for asset in self.assets.values():
                df = asset.get_data(self.timeframe)
                ind_values = ind.calculate_all_sets(df)

                for ind_param_set, ind in ind_values: # Saves each parameter set as a different indicator
                    self.data[f"{asset.name}.{self.timeframe}.{ind.__class__.__name__}.{ind_param_set}"] = ind

        return True

    def _generate_signals(self): # Generate entry signals based on indicators and assets
        return True

    def _backtest_signals(self): # Backtest the generated signals to evaluate performance
        return True

    def calculate_MAE_MFE(self): # Calculate MAE and MFE for each entry signal in the data
        # Checks MAE and MFE for each entry signal in the data, then calculates statistics on them

        return True

    def validate_predictive_power(self): # Validate the predictive power of the indicators and strategies used
        return True

    def run_monte_carlo_simulation(self): # Run Monte Carlo simulations to assess robustness of the findings
        return True

    def _run(self): # Run the edge finder on the provided data
        return True

if __name__ == "__main__":

    # Timeframe and Assets
    op_timeframe = 'M15'
    eurusd = Asset(
        name='EURUSD',
        type='currency_pair',
        market='forex',
        data_path=f'C:\\Users\\Patrick\\Desktop\\Artaxes Portfolio\\MAIN\\MT5_Dados\\Forex',
        timeframe=[op_timeframe])
    assets = {eurusd.name: eurusd}
    
    # Entry Rules L and S    
    entry_long = ((df[asset][tf]['close'] < df[asset][tf][sma]) & (df[asset][tf]['close'].shift(1) > df[asset][tf][sma].shift(1)))
    entry_short = ((df[asset][tf]['close'] > df[asset][tf][sma]) & (df[asset][tf]['close'].shift(1) < df[asset][tf][sma].shift(1)))

    # Indicators
    sma = MA(asset='CURR_ASSET', timeframe=op_timeframe, window=list(range(20, 20+1, 10)), type='sma', price_col='close')

    # Edge Finder Operation
    operation = EdgeFinder(
        EdgeFinderParams(
            name='SMA_Cross',
            indicators={sma.name: sma},
            assets=assets,
            timeframe=op_timeframe,
            entry_signal_rules={'entry_long': entry_long, 'entry_short': entry_short},
            exit_signal_rules=None,
            foward_result_lookback=range(1, 63+1, 1),
            monte_carlo_simulations=1000
        )
    )







