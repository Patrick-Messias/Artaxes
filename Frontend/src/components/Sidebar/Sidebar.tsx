import React, { useState, useEffect } from 'react'
import { api }       from '../../api/client'
import { ModeA }     from './ModeA'
import { ModeB }     from './ModeB'
import { FilterBar } from './FilterBar'
import type { Operation, ParamSet, NavMode, FilterMode, PivotMode } from '../../types'

interface Props {
  fontSize: number
}

const SIDEBAR_WIDTH = 260

export function Sidebar({ fontSize }: Props) {
  const [navMode,    setNavMode]    = useState<NavMode>('A')
  const [pivotMode,  setPivotMode]  = useState<PivotMode>('asset')
  const [filters,    setFilters]    = useState<Set<FilterMode>>(
    new Set(['parset', 'wf', 'selected'])
  )
  const [operations, setOperations] = useState<Operation[]>([])
  const [paramSets,  setParamSets]  = useState<ParamSet[]>([])
  const [loading,    setLoading]    = useState(true)
  const [error,      setError]      = useState<string | null>(null)

  // Load data on mount
  useEffect(() => {
    Promise.all([api.getOperations(), api.getParamSets()])
      .then(([ops, ps]) => {
        console.log("Operações:", ops.length, "ParamSets:", ps.length);
        setOperations(ops)
        setParamSets(ps)
        setLoading(false)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
      })
  }, [])

  const toggleFilter = (f: FilterMode) => {
    setFilters(prev => {
      const next = new Set(prev)
      next.has(f) ? next.delete(f) : next.add(f)
      return next
    })
  }

  return (
    <div
      className="flex flex-col border-r border-[#1E1E1E] bg-[#0A0A0A] shrink-0 overflow-hidden"
      style={{ width: SIDEBAR_WIDTH, fontSize }}
    >

      {/* ── Mode toggle A / B ─────────────────────────────────────── */}
      <div className="flex border-b border-[#1E1E1E] shrink-0">
        {(['A', 'B'] as NavMode[]).map(m => (
          <button
            key={m}
            onClick={() => setNavMode(m)}
            className={`flex-1 py-2 text-[10px] tracking-widest transition-colors ${
              navMode === m
                ? 'text-[#00D9FF] border-b border-[#00D9FF]'
                : 'text-[#333] hover:text-[#666]'
            }`}
          >
            MODE {m}
          </button>
        ))}
      </div>

      {/* ── Mode B pivot selector ─────────────────────────────────── */}
      {navMode === 'B' && (
        <div className="flex gap-1 px-2 py-1.5 border-b border-[#1E1E1E] shrink-0">
          {(['asset', 'strat', 'model'] as PivotMode[]).map(p => (
            <button
              key={p}
              onClick={() => setPivotMode(p)}
              className={`flex-1 py-1 rounded text-[9px] uppercase tracking-wider transition-colors ${
                pivotMode === p
                  ? 'bg-[#00D9FF] text-black font-bold'
                  : 'text-[#333] hover:text-[#666]'
              }`}
            >
              {p}
            </button>
          ))}
        </div>
      )}

      {/* ── Filter bar PS / WF / SEL ──────────────────────────────── */}
      <FilterBar filters={filters} toggle={toggleFilter} />

      {/* ── Tree ─────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto">
        {loading && (
          <div className="p-4 text-[#333] text-[10px] animate-pulse">
            loading...
          </div>
        )}

        {error && (
          <div className="p-4 text-[#FF6B6B] text-[10px]">
            {error}
          </div>
        )}

        {!loading && !error && navMode === 'A' && (
          <ModeA
            operations={operations}
            paramSets={paramSets}
            filters={filters}
          />
        )}

        {!loading && !error && navMode === 'B' && (
          <ModeB
            paramSets={paramSets}
            pivotMode={pivotMode}
            filters={filters}
          />
        )}
      </div>

    </div>
  )
}
