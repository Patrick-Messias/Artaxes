import type { Operation, ParamSet, Asset, EquityPoint, Trade } from '../types'

const BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

// ── Core fetch helpers ────────────────────────────────────────────────────

async function get<T>(path: string, params?: Record<string, string>): Promise<T> {
  const url = new URL(`${BASE}${path}`)
  if (params) {
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== '')
        url.searchParams.set(k, v)
    })
  }
  const res = await fetch(url.toString())
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`)
  return res.json()
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`)
  return res.json()
}

// ── API endpoints ─────────────────────────────────────────────────────────

export interface ParamSetFilters {
  operation?: string
  asset?:     string
  strat?:     string
}

export const api = {

  // Operations
  getOperations: () =>
    get<Operation[]>('/operations'),

  getOperationSummary: (name: string) =>
    get<Record<string, unknown>>(`/operations/${name}/summary`),

  populateOperation: (name: string) =>
    post<{ operation_id: number; message: string }>(
      `/operations/populate/${name}`, {}
    ),

  // Assets
  getAssets: () =>
    get<Asset[]>('/assets'),

  getAsset: (name: string) =>
    get<Asset>(`/assets/${name}`),

  // Param sets — basket selector
  getParamSets: (filters: ParamSetFilters = {}) =>
    get<ParamSet[]>('/results/param-sets', {
      operation: filters.operation ?? '',
      asset:     filters.asset     ?? '',
      strat:     filters.strat     ?? '',
    }),

  // Trades for a specific param_set
  getTrades: (ps_id: number) =>
    get<Trade[]>(`/results/param-sets/${ps_id}/trades`),

  // Equity curve for a basket of param_sets
  getEquityCurve: (ps_ids: number[]) =>
    post<EquityPoint[]>('/results/basket/equity-curve', ps_ids),

  // Health check
  health: () =>
    get<{ status: string }>('/health'),
}
