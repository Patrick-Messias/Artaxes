import pandas as pd, numpy as np, datetime, json
from Model import Model_Parameters, Model
from Asset import Asset, AssetParams, Asset_Portfolio
from Strat import Strat, Strat_Parameters, ExecutionSettings, DataSettings, TimeSettings
from Indicator import Indicator
from Portfolio import Portfolio, Portfolio_Parameters
from Backtest import Backtest, Backtest_Parameters
from Operation import Operation, Operation_Parameters
from ModelMoneyManager import ModelMoneyManager, ModelMoneyManagerParams

# NOTE Should be able to create a Trading Model Portfolio with multiple Strategies taking trades or a simple model that rebalances between a few stocks without "trading"

def test():
    Params = {
        'AT15': {
            'param1': range(4, 80, 4), 
            'param2': range(2, 11, 2),  
            'param3': range(14, 15, 1)  
        }
    }

    # What assets are going to do what enside the Strat
    Asset_Mapping = { 
        'Asset1': {'name': 'CURR_ASSET', 'timeframe': 'M15'}, # CURR_ASSET is the dynamic current Asset in the list being analysed
        'Asset2': {'name': 'EURUSD', 'timeframe': 'D1'} # Defines specific Asset for custom rule
    }

    entry_rules = {
        'entry_long': lambda df: (df['Asset1']['close'] < df['Asset1']['ma']) & (df['Asset1']['close'] < df['Asset1']['low'].shift(1)), # df['Asset2']['close_D1'] > df['Asset2']['ma_D1']
        'entry_short': lambda df: (df['Asset1']['close'] > df['Asset1']['ma']) & (df['Asset1']['close'] > df['Asset1']['high'].shift(1)) # df['Asset2']['close_D1'] < df['Asset2']['ma_D1']
    }
    tf_exit_rules = {
        'tf_exit_long': lambda df: df['Asset1']['close'] > df['Asset1']['ma'],
        'tf_exit_short': lambda df: df['Asset1']['close'] < df['Asset1']['ma']
    }
    sl_exit_rules = None
    tp_exit_rules = None
    be_pos_rules = None
    be_neg_rules = None
    nb_exit_rules = None #Params['AT15']['param3']


    # Usar instâncias das classes
    AT15 = Strat(
        Strat_Parameters(
            name="AT15",
            asset_mapping=Asset_Mapping, # With what will the model define the rules

            execution=ExecutionSettings(order_type='market', offset=0.0),
            data_settings=DataSettings(fill_method='ffill', fillna=0),
            mma_settings=None, # If mma_rules=None then will use default or PMA or other saved MMA define in Operation. Else it creates a temporary MMA with mma_settings
            time_settings=TimeSettings(
                            day_trade=False, 
                            timeTI=None, timeEF=None, timeTF=None, 
                            next_index_day_close=False, friday_close=False, 
                            timeExcludeHours=None, dateExcludeTradingDays=None, dateExcludeMonths=None),
            indicators={
                'ma': Indicator(name='SMA', timeframe=Asset_Mapping['Asset1']['timeframe'], params={'window': list(Params['AT15']['param1'])}, func_path='TA.SMA')
            },
            
            entry_rules=entry_rules,
            tf_exit_rules=tf_exit_rules,
            sl_exit_rules=sl_exit_rules,
            tp_exit_rules=tp_exit_rules,
            be_pos_rules=be_pos_rules,
            be_neg_rules=be_neg_rules,
            nb_exit_rules=nb_exit_rules
        )
    )

    """
    trade=TradeManagementRules( # TradeManagementRules Eliminated, it must check in Strat's generate_signals() with the checks below
        LONG='entry_long' in entry_rules,
        SHORT='entry_short' in entry_rules,
        TF=bool(tf_exit_rules),
        SL=bool(sl_exit_rules),  # Ou definir sl_exit_rules
        TP=bool(tp_exit_rules), # Ou definir tp_exit_rules
        BE_pos=bool(be_pos_rules),
        BE_neg=bool(be_neg_rules),
        NB=list(nb_exit_rules) # Exit by number of bars
    )
    """



    
    model_1 = Model(
        Model_Parameters(
            name='Mean Reversion',
            description='Short term mean reversion strategy',
            assets='FOREX',
            strat={
                'AT15': AT15#, 'AT13': AT13
            },
            execution_timeframe=Asset_Mapping['Asset1']['timeframe'],
            model_money_manager=ModelMoneyManager(MoneyManagerParams(name="AT15_MM"))
        )
    )
    """
    model_2 = Model(
        Model_Parameters(
            name='Trend Following',
            description='Long range trend following using breakout entry',
            assets='FOREX',
            strat={
                'AT20': AT20
            },
            execution_timeframe=Asset_Mapping['Asset2']['timeframe'],
            model_money_manager=ModelMoneyManager(MoneyManagerParams(name="AT20_MM"))
        )
    )
    """
    portfolio = Portfolio(
        Portfolio_Parameters(
            name='Multi_Model_Portfolio',
            models={
                'Model_1': model_1#,
                #'Model_2': model_2
            },
            pma = None, #Portfolio_Manager_Algorithm(PMA_Parameters(name='pma_1')),
            pmm = None #Portfolio_Money_Management(PMM_Parameters(name='pmm_1')),
        )
    )
    backtest = Backtest(
        Backtest_Parameters(

        )
    )
    metrics = {
        'balance': {'total'}
    }

    operation = Operation(
        Operation_Parameters(
            name=f'operation_{datetime.now()}',
            data=portfolio,
            operation=backtest,
            metrics=metrics,
            operation_timeframe='M15',  # Using default timeframe from Asset_Mapping
            date_start=None, date_end=None
        )
    )

    # Durante execução
    results = operation.run()

    # Acesso otimizado
    model_results = results.get_model_results("Model_1")
    strat_results = results.get_strat_results("Model_1", "AT15")

    # Salvar/carregar
    filepath = results.save_results()
    results.load_results(filepath)

    # Serialização para JSON (apenas estrutura, não dados comprimidos)
    json_data = json.dumps(results.to_dict(), indent=2)

def main():
    test()

if __name__ == "__main__":
    main()