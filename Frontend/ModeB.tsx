import React, { useState } from 'react'
import { Row, ParamSetRow } from './TreeRow'
import type { ParamSet, FilterMode, PivotMode } from '../../types'

interface Props {
  paramSets: ParamSet[]
  pivotMode: PivotMode
  filters:   Set<FilterMode>
}

// Maps pivot mode to the primary grouping key
const PIVOT_KEY: Record<PivotMode, keyof ParamSet> = {
  asset: 'asset_name',
  strat: 'strat_name',
  model: 'model_name',
}

// Secondary label — the other two dimensions shown under pivot group
function secondaryLabel(ps: ParamSet, pivotMode: PivotMode): string {
  switch (pivotMode) {
    case 'asset': return `${ps.strat_name} · ${ps.model_name}`
    case 'strat': return `${ps.asset_name} · ${ps.model_name}`
    case 'model': return `${ps.asset_name} · ${ps.strat_name}`
  }
}

export function ModeB({ paramSets, pivotMode, filters }: Props) {
  const [open, setOpen] = useState<Set<string>>(new Set())

  const toggle = (key: string) => setOpen(prev => {
    const next = new Set(prev)
    next.has(key) ? next.delete(key) : next.add(key)
    return next
  })

  const pivotKey = PIVOT_KEY[pivotMode]
  const groups   = [...new Set(paramSets.map(p => String(p[pivotKey])))].sort()

  return (
    <div>
      {groups.map(group => {
        const groupPs    = paramSets.filter(p => String(p[pivotKey]) === group)
        const secondaries = [...new Set(groupPs.map(ps => secondaryLabel(ps, pivotMode)))].sort()

        return (
          <div key={group}>
            {/* Pivot group — Asset / Strat / Model */}
            <Row
              depth={0}
              label={group}
              meta={`${groupPs.length}ps`}
              expanded={open.has(group)}
              onClick={() => toggle(group)}
            />

            {open.has(group) && secondaries.map(sec => {
              const secPs  = groupPs.filter(ps => secondaryLabel(ps, pivotMode) === sec)
              const secKey = `${group}/${sec}`

              return (
                <div key={sec}>
                  {/* Secondary — the other two dimensions */}
                  <Row
                    depth={1}
                    label={sec}
                    meta={`${secPs.length}ps`}
                    expanded={open.has(secKey)}
                    onClick={() => toggle(secKey)}
                  />

                  {/* Param sets */}
                  {open.has(secKey) && secPs.map(ps => (
                    <ParamSetRow
                      key={ps.id}
                      ps={ps}
                      depth={2}
                      filters={filters}
                    />
                  ))}
                </div>
              )
            })}
          </div>
        )
      })}
    </div>
  )
}
