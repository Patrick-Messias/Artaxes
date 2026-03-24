import { create } from 'zustand'
import type { BasketItem, ParamSet } from '../types'

// ── Colors for basket items ───────────────────────────────────────────────
const COLORS = [
  '#00D9FF', '#FF6B6B', '#4ECB71', '#FFD93D', '#C77DFF',
  '#FF9F1C', '#2EC4B6', '#E71D36', '#B5E48C', '#F72585',
  '#4CC9F0', '#F77F00', '#06D6A0', '#EF476F', '#118AB2',
]

// ── Store interface ───────────────────────────────────────────────────────
interface BasketStore {
  items:  BasketItem[]

  // Add a param_set to basket with a specific type
  add:    (ps: ParamSet, type: BasketItem['type']) => void

  // Remove by basket item id
  remove: (id: string) => void

  // Clear all items
  clear:  () => void

  // Check if already in basket
  has:    (ps_id: number, type: BasketItem['type']) => boolean

  // Toggle — adds if not present, removes if present
  toggle: (ps: ParamSet, type: BasketItem['type']) => void
}

// ── Label builder ─────────────────────────────────────────────────────────
function buildLabel(ps: ParamSet, type: BasketItem['type']): string {
  switch (type) {
    case 'wf':
      return `WF · ${ps.asset_name} · ${ps.strat_name}`
    case 'selected':
      return `SEL · ${ps.best_ps_name ?? ps.ps_name} · ${ps.asset_name}`
    case 'parset':
    default:
      return `${ps.ps_name} · ${ps.asset_name}`
  }
}

// ── Zustand store ─────────────────────────────────────────────────────────
export const useBasket = create<BasketStore>((set, get) => ({
  items: [],

  add: (ps, type) => {
    if (get().has(ps.id, type)) return
    const id    = `${ps.id}_${type}`
    const color = COLORS[get().items.length % COLORS.length]
    const label = buildLabel(ps, type)
    set(s => ({
      items: [...s.items, { id, ps_id: ps.id, type, label, ps, color }]
    }))
  },

  remove: (id) =>
    set(s => ({ items: s.items.filter(i => i.id !== id) })),

  clear: () => set({ items: [] }),

  has: (ps_id, type) =>
    get().items.some(i => i.ps_id === ps_id && i.type === type),

  toggle: (ps, type) => {
    if (get().has(ps.id, type)) {
      get().remove(`${ps.id}_${type}`)
    } else {
      get().add(ps, type)
    }
  },
}))
