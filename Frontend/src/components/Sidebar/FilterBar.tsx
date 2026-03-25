import React from 'react'
import type { FilterMode } from '../../types'

interface Props {
  filters: Set<FilterMode>
  toggle:  (f: FilterMode) => void
}

const LABELS: Record<FilterMode, string> = {
  parset:   'PS',
  wf:       'WF',
  selected: 'SEL',
}

const COLORS: Record<FilterMode, string> = {
  parset:   'text-[#00D9FF] border-[#00D9FF33] bg-[#00D9FF11]',
  wf:       'text-[#C77DFF] border-[#C77DFF33] bg-[#C77DFF11]',
  selected: 'text-[#4ECB71] border-[#4ECB7133] bg-[#4ECB7111]',
}

export function FilterBar({ filters, toggle }: Props) {
  return (
    <div className="flex gap-1 px-2 py-1.5 border-b border-[#1E1E1E]">
      {(Object.keys(LABELS) as FilterMode[]).map(f => (
        <button
          key={f}
          onClick={() => toggle(f)}
          className={`flex-1 py-1 rounded text-[9px] uppercase tracking-wider border transition-colors ${
            filters.has(f)
              ? COLORS[f]
              : 'text-[#333] border-transparent hover:text-[#555]'
          }`}
        >
          {LABELS[f]}
        </button>
      ))}
    </div>
  )
}
