// ── Navigation & Filter types ─────────────────────────────────────────────
export type NavMode    = 'A' | 'B'
export type PivotMode  = 'asset' | 'strat' | 'model'
export type FilterMode = 'parset' | 'wf' | 'selected'

// ── API response types ────────────────────────────────────────────────────
export interface Operation {
  id:           number
  name:         string
  date_start:   string | null
  date_end:     string | null
  status:       string
  n_models:     number
  n_strats:     number
  n_assets:     number
  n_param_sets: number
  created_at:   string | null
}

export interface ParamSet {
  id:              number
  ps_name:         string
  param_json:      Record<string, unknown> | null
  n_trades:        number | null
  total_pnl:       number | null
  trades_path:     string | null
  pnl_matrix_path: string | null
  strat_name:      string
  backtest_mode:   string | null
  model_name:      string
  exec_tf:         string | null
  asset_name:      string
  market:          string | null
  operation_name:  string
  date_start:      string | null
  date_end:        string | null
  best_ps_name:    string | null
  best_wfe:        number | null
  wf_json_path:    string | null
}

export interface Asset {
  id:           number
  name:         string
  type:         string | null
  market:       string | null
  tick:         number | null
  tick_fin_val: number | null
  lot_min:      number | null
  lot_step:     number | null
  lot_max:      number | null
  margin_rate:  number | null
  swap_long:    number | null
  swap_short:   number | null
  data_path:    string | null
  created_at:   string | null
}

// ── Basket types ──────────────────────────────────────────────────────────
export interface BasketItem {
  id:     string                          // unique: `${ps_id}_${type}`
  ps_id:  number
  type:   'parset' | 'wf' | 'selected'
  label:  string
  ps:     ParamSet
  color:  string
}

// ── Chart types ───────────────────────────────────────────────────────────
export interface EquityPoint {
  datetime:       string
  total_pnl:      number
  cumulative_pnl: number
  [key: string]:  number | string         // ps_{id} columns
}

export interface Trade {
  entry_price:    number
  exit_price:     number
  lot_size:       number
  profit:         number
  entry_datetime: string
  exit_datetime:  string
  exit_reason:    string
  asset:          string
}
