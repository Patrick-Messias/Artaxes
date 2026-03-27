import polars as pl, itertools, sys
sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend')
sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend\indicators')

from Strat import Strat, StratParams, ExecutionSettings # type: ignore
from StratMoneyManager import StratMoneyManager, StratMoneyManagerParams # type: ignore
from Walkforward import Walkforward # type: ignore
from Indicator import Indicator # type: ignore

from MA import MA # type: ignore
from VAR import VAR # type: ignore
from ATR_SL import ATR_SL # type: ignore
from RawData import RawData # type: ignore
from PriorCote import PriorCote # type: ignore
from DayOpen import DayOpen # type: ignore

def strat_template():
    strat_execution_timeframe="M15"
    _strat_name = "AT15"

    _strat_walkforward = Walkforward(
        wfm_configs=[[is_len, os_len, os_len] for is_len, os_len in itertools.product([1, 2, 4, 12, 16, 24, 36, 48], [1, 2, 4, 12, 16, 24, 36, 48])],
        wfm_is_always_higher_or_equal_to_oos=True,
        matrix_resolution='weekly', time_mode = 'calendar_days',
        is_metric='pnl', is_top_n=1, is_logic='highest', is_order='des',
        wf_selection_metric='wfe', wf_selection_analysis_radius_n=1,
        wf_selection_logic='highest_stable', wf_returns_mode='selected'
    )

    _strat_execution_settings = ExecutionSettings(
        hedge=True, strat_num_pos=[1,1], strat_max_num_pos_per_day=[999,999],
        order_type='market', limit_order_base_calc_ref_price='open', 
        slippage=0.0, commission=0.0, # * Tick 
        day_trade=True, timeTI=None, timeEF=None, timeTF=None, next_index_day_close=False, # "0:00"
        day_of_week_close_and_stop_trade=[], timeExcludeHours=None, dateExcludeTradingDays=None, dateExcludeMonths=None, 
        fill_method='ffill', fillna=0, trade_pnl_resolution='daily', 
        backtest_mode="ohlc", convert_sltp_to_pct=False, print_logs=False
    )

    _strat_money_manager = StratMoneyManager(StratMoneyManagerParams(
        sizing_method="neutral", capital_method="fixed", compound_fract=1.0, dist_fixed=None,
        sizing_params={"fixed_lot": 1.0, "risk_pct": 0.01, "risk_pct_min": 0.001, "risk_pct_max": 0.05,"pct": 0.01,"kelly_weight": 0.25, "var_confidence": 0.95, "min_trades": 30}
    )) # If mma_rules=None then will use default or PMA or other saved MMA define in Operation. Else it creates a temporary MMA with mma_settings

    _strat_param_sets = {
        #====================================================||
        'execution_tf': strat_execution_timeframe,
        'backtest_start_idx': 21,
        'limit_order_exclusion_after_period': 1,
        'limit_order_perc_treshold_for_order_diff': 0.03,
        'limit_can_enter_at_market_if_gap': False,
        'limit_opposite_order_closes_pending': False,

        'exit_nb_only_if_pnl_is': 0, 
        'exit_nb_long': range(0, 0+1, 0),
        'exit_nb_short': range(0, 0+1, 0),
        #====================================================||
        'sl_perc': range(3, 10+1, 4), 
        'tp_perc': range(3, 10+1, 4), 

        'param1': range(21, 252+1, 21), 
        'param2': range(8, 24+1, 8), 
        'param3': ['sma', 'sma'] 
        #====================================================||
    }

    # User imput Indicators
    _strat_ind_pool = { 
        'atr': ATR_SL(asset=None, timeframe=strat_execution_timeframe, window='param2'),
        'var': VAR(asset=None, timeframe=strat_execution_timeframe, window='param2', alpha=0.01, var_type='parametric', price_col='close'),
        'ema': MA(asset=None, timeframe=strat_execution_timeframe, window='param1', ma_type='param3', price_col='close'),
    }

    def _strat_signals(df: pl.DataFrame, params: dict) -> dict:
        # Can use columns df['high'] or str 'high' to point

        atr = df['atr'].shift(1)
        var = df['var'].shift(1)

        bull = df['close'].shift(1) < df['open'].shift(1)
        bear = df['close'].shift(1) > df['open'].shift(1)

        entry_long  = bull.shift(1) & bull.shift(2) & bull.shift(3) #& (ema < ema.shift(1))
        entry_short = bear.shift(1) & bear.shift(2) & bear.shift(3) #& (ema > ema.shift(1))

        exit_tf_long  = bear.shift(1) & bear.shift(2)
        exit_tf_short = bull.shift(1) & bull.shift(2)

        # Preço da ordem pendente
        limit_long_price  = df['open'].shift(1) #'high[1]' #
        limit_short_price = df['open'].shift(1) #'low[1]' #

        # Distâncias (definidas ANTES de serem usadas)
        sl_long_price  = limit_long_price - atr * params['sl_perc'] 
        sl_short_price = limit_long_price + atr * params['sl_perc'] 

        # TP absoluto: 2R
        tp_long_price  = limit_long_price + atr  * params['tp_perc']
        tp_short_price = limit_long_price - atr * params['tp_perc']

        # Trailing: 0.5R
        trail_long_dist  = None #sl_long_dist  * 0.5
        trail_short_dist = None #sl_short_dist * 0.5

        # BE: distância de 1R para ativar (C++ faz: price >= entry + be_dist)
        be_long_dist  = None #sl_long_dist  * 1.0
        be_short_dist = None #sl_short_dist * 1.0

        # Position Sizing
        var = df['var'].abs().clip(lower_bound=0.0001)  # evita divisão por zero
        lot_series = (pl.lit(0.01) / var.shift(1)).clip(0.1, 10.0)

        # n = len(df)
        # compound_fract_vals = np.ones(n, dtype=np.float64)
        # compound_fract_vals[10000:] = 0.1
        # compound_fract_series = pl.Series("compound_fract_series", compound_fract_vals)
        compound_fract_series = None
        dist_signal_ref = None
        
        return {
            '__sig_key_params': ['param1', 'param2', 'param3', 'sl_perc', 'tp_perc'],

            'entry_long':       entry_long,
            'entry_short':      entry_short,
            'exit_long':        exit_tf_long,
            'exit_short':       exit_tf_short,

            'sl_price_long':    sl_long_price,
            'sl_price_short':   sl_short_price,
            'tp_price_long':    tp_long_price,
            'tp_price_short':   tp_short_price,

            'limit_long':       limit_long_price,
            'limit_short':      limit_short_price,

            'trail_long':       trail_long_dist,
            'trail_short':      trail_short_dist,

            'be_trigger_long':  be_long_dist,
            'be_trigger_short': be_short_dist,

            'custom_lot_size_long':  lot_series,
            'custom_lot_size_short': lot_series,
            'compound_fract_series': compound_fract_series,
            'dist_signal_ref': dist_signal_ref,
        }








    try:
        strat = Strat(
            StratParams(
                name=_strat_name,
                operation=_strat_walkforward,
                execution_settings=_strat_execution_settings,
                strat_money_manager=_strat_money_manager,
                params=_strat_param_sets, 
                indicators=_strat_ind_pool,
                signals=_strat_signals
            )
        )
    except Exception as e:
        print(f"< [Error] Generating Strat, {e}")

    return strat






















