import pandas as pd, numpy as np, datetime, json, sys
from Model import ModelParams, Model
from Asset import Asset, AssetParams, Asset_Portfolio
from Strat import Strat, StratParams, ExecutionSettings, DataSettings, TimeSettings
from Portfolio import Portfolio, PortfolioParams
from Backtest import Backtest, BacktestParams
from Operation import Operation, OperationParams
from ModelMoneyManager import ModelMoneyManager, ModelMoneyManagerParams
from StratMoneyManager import StratMoneyManager, StratMoneyManagerParams
from ModelSystemManager import ModelSystemManager, ModelSystemManagerParams

sys.path.append(r'C:\Users\Patrick\Desktop\ART_Backtesting_Platform\Backend\Indicators')
from MA import MA # type: ignore

# NOTE Should be able to create a Trading Model Portfolio with multiple Strategies taking trades or a simple model that rebalances between a few stocks without "trading"

def test():
    Params = {
        'AT15': {
            'param1': range(10, 10+1, 10), 
            'param2': range(5, 5+1, 5),  
            'param3': range(14, 14+1, 1)  
        }
    }

    execution_tf='M15'
    eurusd_daily = Asset(
        name='EURUSD',
        type='currency_pair',
        market='forex',
        data_path=f'C:\\Users\\Patrick\\Desktop\\Artaxes Portfolio\\MAIN\\MT5_Dados\\Forex',
        timeframe=['D1']
    )
    eurusd = Asset(
        name='EURUSD',
        type='currency_pair',
        market='forex',
        data_path=f'C:\\Users\\Patrick\\Desktop\\Artaxes Portfolio\\MAIN\\MT5_Dados\\Forex',
        timeframe=[execution_tf]
    )
    gbpusd = Asset(
        name='GBPUSD',
        type='currency_pair',
        market='forex',
        data_path=f'C:\\Users\\Patrick\\Desktop\\Artaxes Portfolio\\MAIN\\MT5_Dados\\Forex',
        timeframe=[execution_tf]
    )
    usdjpy = Asset(
        name='USDJPY',
        type='currency_pair',
        market='forex',
        data_path=f'C:\\Users\\Patrick\\Desktop\\Artaxes Portfolio\\MAIN\\MT5_Dados\\Forex',
        timeframe=[execution_tf]
    )

    Model_Assets = Asset_Portfolio({
        'name': 'forex',
        'assets': {eurusd.name: eurusd, gbpusd.name: gbpusd, usdjpy.name: usdjpy}
    })
    #Forex = {eurusd.name: eurusd, gbpusd.name: gbpusd, usdjpy.name: usdjpy} # Criar uma def que retorna o(s) asset com o timeframe que quer
    Strat_Assets = {'eurusd': eurusd_daily} # Assets de suporte seja para calcular regras ou indicadores

    entry_rules = {
        'entry_long': lambda df: (
            df['CURR_ASSET'][execution_tf]['close'] < df['CURR_ASSET'][execution_tf]['sma']) 
            & (df['CURR_ASSET'][execution_tf]['close'].shift(1) < df['CURR_ASSET'][execution_tf]['low'].shift(1)) 
            & (df['eurusd']['D1']['close'] > df['eurusd']['D1']['close'].shift(1)
            ), 
        'entry_short': lambda df: (
            df['CURR_ASSET'][execution_tf]['close'] > df['CURR_ASSET'][execution_tf]['sma']) 
            & (df['CURR_ASSET'][execution_tf]['close'].shift(1) > df['CURR_ASSET'][execution_tf]['high'].shift(1)) 
            & (df['eurusd']['D1']['close'] < df['eurusd']['D1']['close'].shift(1)
            ) 
    }
    tf_exit_rules = {
        'tf_exit_long': lambda df: (df['CURR_ASSET'][execution_tf]['close'] > df['CURR_ASSET'][execution_tf]['sma']),
        'tf_exit_short': lambda df: (df['CURR_ASSET'][execution_tf]['close'] < df['CURR_ASSET'][execution_tf]['sma'])
    }
    sl_exit_rules = None
    tp_exit_rules = None
    be_pos_rules = None
    be_neg_rules = None
    nb_exit_rules = None #Params['AT15']['param3']


    # Usar instâncias das classes
    AT15 = Strat(
        StratParams(
            name="AT15",
            strat_support_assets=Strat_Assets, # With what will the model define the rules

            execution_settings=ExecutionSettings(order_type='market', offset=0.0),
            data_settings=DataSettings(fill_method='ffill', fillna=0),
            mma_settings=None, # If mma_rules=None then will use default or PMA or other saved MMA define in Operation. Else it creates a temporary MMA with mma_settings
            time_settings=TimeSettings(
                            day_trade=False, 
                            timeTI=None, timeEF=None, timeTF=None, 
                            next_index_day_close=False, friday_close=False, 
                            timeExcludeHours=None, dateExcludeTradingDays=None, dateExcludeMonths=None),
            indicators={  
                'ma': MA( 
                        asset='CURR_ASSET', # if 'CURR_ASSET' then must create for all assets in Model's Assets
                        timeframe=execution_tf,
                        window=list(Params['AT15']['param1']),
                        ma_type='sma',
                        price_col='close'
                    )
            },

            #Mudar onde for relevante, na definição do ind (acima) deve usar list para colocar >1 parametro
            #ITERAR SOBRE models.assets -> strat -> ind.calculate_all_sets
            
            entry_rules=entry_rules,
            tf_exit_rules=tf_exit_rules,
            sl_exit_rules=sl_exit_rules,
            tp_exit_rules=tp_exit_rules,
            be_pos_rules=be_pos_rules,
            be_neg_rules=be_neg_rules,
            nb_exit_rules=nb_exit_rules,

            strat_money_manager=StratMoneyManager(StratMoneyManagerParams(name="AT15_SMM"))
        )
    )

    """
    trade=TradeManagementRules( # TradeManagementRules vars Eliminated, it must check in Strat's generate_signals() with the checks below
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
        ModelParams(
            name='Mean Reversion',
            description='Short term mean reversion strategy',
            assets=Model_Assets, # CURR_ASSET refers to this one in strat_support_assets
            strat={
                'AT15': AT15#, 'AT13': AT13
            },
            execution_timeframe=execution_tf,
            model_money_manager=ModelMoneyManager(ModelMoneyManagerParams(name="Model1_MM")),
            model_system_manager=None  # Optional - will use default system management
        )
    )
    """
    model_2 = Model(
        ModelParams(
            name='Trend Following',
            description='Long range trend following using breakout entry',
            assets=Model_Assets,
            strat={
                'AT20': AT20
            },
            execution_timeframe=execution_tf,
            model_money_manager=ModelMoneyManager(ModelMoneyManagerParams(name="Model2_MM")),
            model_system_manager=None  # Optional - will use default system management
        )
    )
    """
    portfolio = Portfolio(
        PortfolioParams(
            name='Multi_Model_Portfolio',
            models={
                'Model_1': model_1#,
                #'Model_2': model_2
            },
            portfolio_money_manager = None, #Portfolio_Manager_Algorithm(PMA_Parameters(name='pma_1')),
            portfolio_system_manager = None #Portfolio_Money_Management(PMM_Parameters(name='pmm_1')),
        )
    )
    backtest = Backtest()
    metrics = {
        'balance': {'total'}
    }

    operation = Operation(
        OperationParams(
            name=f'operation_{datetime.datetime.now()}',
            data=portfolio,
            operation=backtest,
            metrics=metrics,
            operation_timeframe=execution_tf,  # Using default timeframe from strat_support_assets
            date_start=None, date_end=None
        )
    )

    # Durante execução
    results = operation.run()
    """ WIP ABOVE, ADD results class for operation run
    # Acesso otimizado
    model_results = results.get_model_results("Model_1")
    strat_results = results.get_strat_results("Model_1", "AT15")

    # Salvar/carregar
    filepath = results.save_results()
    results.load_results(filepath)

    # Serialização para JSON (apenas estrutura, não dados comprimidos)
    json_data = json.dumps(results.to_dict(), indent=2)
    """
def main():
    test()

if __name__ == "__main__":
    main()