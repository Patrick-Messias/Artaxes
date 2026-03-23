-- ═══════════════════════════════════════════════════════════════════════════
-- DuckDB Schema
-- ═══════════════════════════════════════════════════════════════════════════
-- DuckDB reads Parquet files directly via read_parquet() without copying data
-- All tables below are Metadata, hard data is located in disk
-- ═══════════════════════════════════════════════════════════════════════════

-- ── ASSET_GROUPS ─────────────────────────────────────────────────────────────
-- Shared parameters for groups of assets (ex: all SP500 stocks share same tick)
-- Asset inherits group parameters when its own field is NULL
CREATE TABLE IF NOT EXISTS asset_groups (
    id              INTEGER PRIMARY KEY,
    name            VARCHAR NOT NULL,           -- ex: 'SP500', 'Forex Majors'
    market          VARCHAR,
    tick            DOUBLE,
    tick_fin_val    DOUBLE,
    lot_min         DOUBLE,
    lot_step        DOUBLE,
    lot_max         DOUBLE,
    margin_rate     DOUBLE
);

-- ── ASSETS ───────────────────────────────────────────────────────────────────
-- Technical data of each Asset
-- If group_id is set, inherits NULL fields from asset_groups
CREATE TABLE IF NOT EXISTS assets (
    id              INTEGER PRIMARY KEY,
    group_id        INTEGER REFERENCES asset_groups(id),  -- optional group inheritance
    name            VARCHAR NOT NULL,
    type            VARCHAR,
    market          VARCHAR,
    tick            DOUBLE,
    tick_fin_val    DOUBLE,
    lot_min         DOUBLE,
    lot_step        DOUBLE,
    lot_max         DOUBLE,
    margin_rate     DOUBLE,
    swap_long       DOUBLE,
    swap_short      DOUBLE,
    data_path       VARCHAR,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- ── OPERATIONS ───────────────────────────────────────────────────────────────
-- Every time that the user does a full backtest it turns into an operation
-- This is the container of everything that has been calculated in that round
CREATE TABLE IF NOT EXISTS operations (
    id              INTEGER PRIMARY KEY,        
    name            VARCHAR NOT NULL UNIQUE,    -- ex: 'op_eurusd_2020'
    date_start      DATE,                       -- backtest period start
    date_end        DATE,                       -- backtest period end
    op_timeframe    VARCHAR,                    -- operation's main timeframe
    status          VARCHAR DEFAULT 'pending',  -- pending | running | done | erro
    results_path    VARCHAR,                    -- parquet results base path: results/{name}/ 
    create_at       TIMESTAMP DEFAULT NOW(),   
    finished_at     TIMESTAMP
);

-- ── MODELS ───────────────────────────────────────────────────────────────────
-- Enside ever operation, models defined by user
-- Models group Strat(s) and Asset(s) 
CREATE TABLE IF NOT EXISTS models (
    id              INTEGER PRIMARY KEY,
    operation_id    INTEGER NOT NULL REFERENCES operations(id),
    name            VARCHAR NOT NULL,   -- ex: 'MA Trend Following'
    exec_tf         VARCHAR             -- execution timeframe: 'M15', 'H1', etc
);

-- ── MODEL_ASSETS ─────────────────────────────────────────────────────────────
-- Junction table: which assets each model uses
-- A model can have multiple assets; an asset can appear in multiple models
CREATE TABLE IF NOT EXISTS model_assets (
    model_id        INTEGER NOT NULL REFERENCES models(id),
    asset_id        INTEGER NOT NULL REFERENCES assets(id),
    PRIMARY KEY (model_id, asset_id)
);

-- ── STRATS ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS strats (
    id                      INTEGER PRIMARY KEY,                        
    model_id                INTEGER NOT NULL REFERENCES models(id),
    name                    VARCHAR NOT NULL,                           -- ex: 'AT15'
    backtest_mode           VARCHAR DEFAULT 'ohlc',                     -- ohlc | open | close | open-open | close-close | avg_price | ohlc_m1 | tick
    order_type              VARCHAR DEFAULT 'market',
    day_trade               BOOLEAN DEFAULT FALSE,
    hedge                   BOOLEAN DEFAULT FALSE,
    convert_sltp_to_pct     BOOLEAN DEFAULT FALSE,
    exec_settings_json      JSON                                        -- complete serialized settings
);

-- ── PARAM_SETS ───────────────────────────────────────────────────────────────
-- Every combination of parameters worked in Strat/Asset
-- Points to parquets in disk, DuckDB read directly without copying data
CREATE TABLE IF NOT EXISTS param_sets (
    id                      INTEGER PRIMARY KEY,
    strat_id                INTEGER NOT NULL REFERENCES strats(id),
    asset_id                INTEGER NOT NULL REFERENCES assets(id),
    name                    VARCHAR NOT NULL,                           -- ex: 'param_set-8-2'
    param_json              JSON,                                       -- dict of parameters: {param2: 8, sl_perc: 2}
    trades_path             VARCHAR,                                    -- parquet path of individual trades
    pnl_matrix_path         VARCHAR,                                    -- parquet path for pnl_matrix
    lot_matrix_path         VARCHAR,                                    -- parquet path for lot_matrix
    n_trades                INTEGER,                                    -- total trades (cache with quick queries)
    total_pnl               DOUBLE,                                     -- PnL total % (cache)
    created_at              TIMESTAMP DEFAULT NOW()
);

-- ── WF_RESULTS ───────────────────────────────────────────────────────────────
-- Walkforward results reference by Strat/Asset
-- Doesn't copy JSON, only saves path for reading on demmand
CREATE TABLE IF NOT EXISTS wf_results (
    id                      INTEGER PRIMARY KEY,
    strat_id                INTEGER NOT NULL REFERENCES strats(id),
    asset_id                INTEGER NOT NULL REFERENCES assets(id),
    json_path               VARCHAR NOT NULL,                           -- path to all_wf_results.json
    matrix_resolution       VARCHAR,
    time_mode               VARCHAR,
    is_metric               VARCHAR,
    is_top_n                INTEGER,
    is_logic                VARCHAR,
    is_order                VARCHAR,
    best_ps_name            VARCHAR,                                    -- param_set selected by WF
    best_wfe                DOUBLE,                                     -- best WFE (quick querry cache)
    wf_select_metric        VARCHAR,                                    -- metric used 'wfe', 'pnl', etc
    wf_selection_analysis_radius_n  INTEGER,
    wf_select_logic         VARCHAR,
    created_at              TIMESTAMP DEFAULT NOW()
);

-- ── PORTFOLIO_SIMULATION ─────────────────────────────────────────────────────
-- Every portfolio simulation created by the user
-- Can combie param_set from multiple operations
CREATE TABLE IF NOT EXISTS portfolio_simulation(
    id                      INTEGER PRIMARY KEY,
    name                    VARCHAR NOT NULL,                           -- ex: 'sim_forex_2023'
    operation_ids           INTEGER[],                                  -- operation_ids array included
    config_json             JSON,                                       -- config of the simulation (capital, MM, etc)
    status                  VARCHAR DEFAULT 'pending',
    created_at              TIMESTAMP DEFAULT NOW(),
    finished_at             TIMESTAMP
);

-- ── PORTFOLIO_POSITIONS ──────────────────────────────────────────────────────
-- Detailed book of every position enside a simulation
-- Registers everything of every trade simulated in portfolio
CREATE TABLE IF NOT EXISTS portfolio_position (
    id                      BIGINT PRIMARY KEY,
    sim_id                  INTEGER NOT NULL REFERENCES portfolio_simulation(id),
    datetime                TIMESTAMP NOT NULL,                         -- Entry moment
    asset_id                INTEGER REFERENCES assets(id),
    model_name              VARCHAR,
    strat_name              VARCHAR,
    ps_name                 VARCHAR,                                    -- what parset generated entry
    side                    VARCHAR,                                    -- 'long' | 'short'
    lot_size                DOUBLE,
    entry_price             DOUBLE,
    exit_price              DOUBLE,
    stop_loss               DOUBLE,
    take_profit             DOUBLE,
    exit_datetime           TIMESTAMP,
    exit_reason             VARCHAR,                                    -- 'TP' | 'SL' | 'TF' | 'EF' etc
    pnl                     DOUBLE                                      -- results in %
);

-- ── PORTFOLIO_DAILY_RESULTS ──────────────────────────────────────────────────
-- Light version with only [datetime, pnl_total] for quick calculations
-- Equity curve, drawdown, sharpe, etc; Everything runs in SQL with this table
-- without needing to open detailed positions
CREATE TABLE IF NOT EXISTS portfolio_daily_results (
    id                      BIGINT PRIMARY KEY,
    sim_id                  INTEGER NOT NULL REFERENCES portfolio_simulation(id),
    datetime                TIMESTAMP NOT NULL,
    total_pnl               DOUBLE,                                     -- sum of all PnL of the day
    n_positions             INTEGER,                                    -- how many positions where opened
    cumulative_pnl          DOUBLE                                      -- PnL summed up to this point (equity curve)
);

-- ═══════════════════════════════════════════════════════════════════════════
-- VIEWS — atalhos para queries frequentes no frontend
-- ═══════════════════════════════════════════════════════════════════════════

-- Lists all operations with count of Models/Strat/Assets
CREATE VIEW IF NOT EXISTS v_operations_summary AS
SELECT
    o.id,
    o.name,
    o.date_start,
    o.date_end,
    o.status,
    COUNT(DISTINCT m.id)  AS n_models,
    COUNT(DISTINCT s.id)  AS n_strats,
    COUNT(DISTINCT ma.asset_id) AS n_assets,
    COUNT(DISTINCT ps.id) AS n_param_sets,
    o.created_at
FROM operations o
LEFT JOIN models m        ON m.operation_id = o.id
LEFT JOIN strats s        ON s.model_id = m.id
LEFT JOIN model_assets ma ON ma.model_id = m.id
LEFT JOIN param_sets ps   ON ps.strat_id = s.id
GROUP BY o.id, o.name, o.date_start, o.date_end, o.status, o.created_at;


-- Lista all param_sets with path of the parquets and metrics
-- Frontend uses this to create the selector basket
CREATE VIEW IF NOT EXISTS v_param_sets_full AS
SELECT
    ps.id,
    ps.name          AS ps_name,
    ps.param_json,
    ps.n_trades,
    ps.total_pnl,
    ps.trades_path,
    ps.pnl_matrix_path,
    s.name           AS strat_name,
    s.backtest_mode,
    m.name           AS model_name,
    m.exec_tf,
    a.name           AS asset_name,
    a.market,
    o.name           AS operation_name,
    o.date_start,
    o.date_end,
    wf.best_ps_name,
    wf.best_wfe,
    wf.json_path     AS wf_json_path
FROM param_sets ps
JOIN strats s       ON s.id = ps.strat_id
JOIN models m       ON m.id = s.model_id
JOIN operations o   ON o.id = m.operation_id
JOIN assets a       ON a.id = ps.asset_id
LEFT JOIN wf_results wf ON wf.strat_id = s.id AND wf.asset_id = ps.asset_id;


-- Equity curve of a simulation of portfolio
-- Uses: SELECT * FROM v_portfolio_equity WHERE sim_id = 1
CREATE VIEW IF NOT EXISTS v_portfolio_equity AS
SELECT
    sim_id,
    datetime,
    total_pnl,
    cumulative_pnl,
    n_positions,
    -- Drawdown calculado via window function
    cumulative_pnl - MAX(cumulative_pnl) OVER (
        PARTITION BY sim_id
        ORDER BY datetime
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS drawdown
FROM portfolio_daily_results
ORDER BY sim_id, datetime;
















