


060326 - Basic data handling, indicator, signal, py-cpp-py framwork, backtesting, walkforward done
060326 - New branch created to rework redundant indicador data being transfered to cpp and changing JSON to MessagePack for data transfer
070326 - Develop new indicator handling and MessagePack to transfer data py-cpp, planning on simplifying signals on python to make more beginner \
friendly, now cpp only reads ready lists of bool.
110326 - Consolidating and optimizing signal generation method
170326 - Optimization Done:
    # Optimizations Done:
    # - Removed copy of data for backtest in cpp, now backtest receives base_data and sim_data
    # - Separate conversion of JSON only one time and saves in cpp cache so it doesn't conver for each backtest
    # - Simuation only points to signals and indicators of which that simulation needs 

    # - replaced msgpack to py::array_t 
    # - bar_dates/times/days computed only 1x in Engine
    # - trade_to_pydict direct without nlohmann on cpp-py 
    
    # - Signals optimization
    # - Bool (Entry/Exit_TF)
    # - Signal Refs (TP/SL/Trail/Limit/BreakEven) only points to OHLC+Indicators present during each parset backtest 
    # - Making shure still can freely calculate new columns in signal generation

    # - daily_results_matrix reallocs eliminated from the biggest vector of the loop
    # - Trade without std::optinal, entry_datetime/exit_datetime with char[20]
    # - Trade -> py::dict direct with bindings.cpp
    # - operation.cpp without trades_to_json, trades go directly as vector<Trade>, zero serialization
    # - thread_local_rng in generate_id
    # - unordered_set for closed_days, lookup O(1) instead of std::find O(n)

    # - in engine_result_to_pydict instead of list[dict] by trade, returns numpy array by column

    # - _save_tradaes: sim_slice, instead of converting to list[dict], save pl.DataFrame directly by sim
    # - _operation: keep int_pool_arrays as np.array
    # - _report_pnl_summary and _plot_pnl_curves: works with DataFrame

    # - Implementing batch system with current defs
    # - Reimplementing simple column call 'high' instead of full dataframe df['high']
    # - wfm by cumm batch, instead of x wfm_columnar, just join into one pivot
    # - Redundant strat_signals calls by not calculating equal signals


  
