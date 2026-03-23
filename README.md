# XXX - Recriar ponte py - cpp - py
# XXX - Recriar sistema de regras para ficar mais simples (py gera sinal - cpp executa)
# XXX - Optimization 

# XXX - Existe um problema no updated pnl daily, se eu considero um novo parset com trade comprado ainda vou estar simulando a variação baseada na abertura, como tratar? \
#talvez colocar que se trocou o parset e o parset novo já tem trade aberto ele considera a variação do pct_change e não do (close-open)/open, logo qualquer nova variação negativa -, positiva +
# XXX - Adicionar lado, WFM que pode selecionar optmizize LONG, SHORT or BOTH sides tanto em WFM quanto Portfolio Simulator. Redundante salvar lado/asset/model/strat, se orientar pelo _results_map
# XXX - Desenvolver MM sistema de slippage, lot, comission, etc; Tanto em py tanto cpp, Model lida com Asset

# XXX - Substituir dist_signal_ref e dist_fixed por uma serie opcional de quanto seria a distância para considerar o calculo, para caso precise e não tenha SL
# XXX - Corrigir bug compound_fract_series

# XXX - Develop start_date - end_date for operation
# XXX - Create method to save results and clear _results_map

# XXX - Modernize Classes
# XXX - Adicionar novo Backtester para Close-Close, Open-Open.

# - Create backend and frontend for results panel
#   - Schema DuckDB: Metadata tables + views over all existing parquets
#   - FastAPI: Endpoints REST to list and consult results, WebSocket for streaming of backtest progress
#   - React + TS: Frontend consumes API, selected results basket, plots graphs
# - Panel with all model-strat-asset-parest/wf results, can filter between all, select analysis, plots, and CRUD Portfolio with results, for Portfolio Simulation 

# - SM and MM for Portfolio and Models

# - Dev Roadmap png/list 
# - Deselop SM selection system for Models/Strats/Assets
# - Develop Portfolio Simulator
# - If Portfolio Simulation then Slippage and Commission on backtest = 0 and calculates on lot_size of Portfolio

# - Modify HTF-LTF function to work from LTF to HTF also
# - Monte Carlo

# - Parquet parset save for very high ammount of parsets, for >2000 parsets per strat freezes
# - Option to save and use only Trade or daily returns

# - Add M1 Backtest (Converts current datetime into M1 signals if asset's data is available) 
# - Vectorized and not vectorized [i] backtest
# - Implement tick backtest as data and timeframe [1 tick, 5, 20, etc] for calculations

# - Candlestick chart with trade entry validation

# - Best leave for PS? Scale in/out of open trade, can use entry/exit signals or new specific signals, must be able to handle market or limit/stop orders, how will this handle in Portfolio Simulator?
# - Update position in backtest (for multiple entries) must have daily returns update with lot_size update
# PortfolioSimulator deve ter a opção de ter uma matrix de covariancia para models uma para strats e uma para assets? talvez uma que armazene as posições selecionadas apenas?


"""
