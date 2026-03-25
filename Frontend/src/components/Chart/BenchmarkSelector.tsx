import React, { useRef } from 'react'
import type { EquityPoint } from '../../types'

interface Props {
  onSelect:     (data: EquityPoint[]) => void
  onClear:      () => void
  hasBenchmark: boolean
  asPerc:       boolean
  onTogglePerc: () => void
}

export function BenchmarkSelector({
  onSelect, onClear, hasBenchmark, asPerc, onTogglePerc
}: Props) {
  const fileRef = useRef<HTMLInputElement>(null)

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = ev => {
      const text   = ev.target?.result as string
      const lines  = text.trim().split('\n')
      const header = lines[0].split(',').map(h => h.trim().toLowerCase())

      // Auto-detect datetime and price columns
      const dtIdx = header.findIndex(h =>
        h.includes('datetime') || h.includes('date') || h.includes('time')
      )
      const prIdx = header.findIndex(h =>
        h === 'close' || h === 'price' || h === 'value'
      ) !== -1
        ? header.findIndex(h => h === 'close' || h === 'price' || h === 'value')
        : 1  // fallback to second column

      const rows = lines.slice(1)
        .map(l => l.split(','))
        .filter(cols => cols.length > prIdx)
        .map(cols => ({
          datetime: cols[dtIdx]?.trim() ?? '',
          price:    parseFloat(cols[prIdx]?.trim() ?? '0'),
        }))
        .filter(r => !isNaN(r.price) && r.datetime)

      if (rows.length === 0) return

      const base = rows[0].price
      const points: EquityPoint[] = rows.map((r, i) => ({
        datetime:       r.datetime,
        total_pnl:      i === 0 ? 0
          : ((r.price - rows[i - 1].price) / rows[i - 1].price) * 100,
        cumulative_pnl: ((r.price - base) / base) * 100,
      }))

      onSelect(points)
    }
    reader.readAsText(file)

    // Reset input so same file can be re-selected
    e.target.value = ''
  }

  return (
    <div className="flex items-center gap-2">
      {hasBenchmark ? (
        <button
          onClick={onClear}
          className="text-[9px] px-2 py-1 border border-[#FFD93D33] rounded text-[#FFD93D] hover:border-[#FF6B6B33] hover:text-[#FF6B6B] transition-colors"
        >
          − benchmark
        </button>
      ) : (
        <button
          onClick={() => fileRef.current?.click()}
          className="text-[9px] px-2 py-1 border border-[#1E1E1E] rounded text-[#444] hover:text-[#FFD93D] hover:border-[#FFD93D33] transition-colors"
        >
          + benchmark
        </button>
      )}

      {/* as % toggle */}
      <label className="flex items-center gap-1 cursor-pointer">
        <div
          onClick={onTogglePerc}
          className={`w-6 h-3 rounded-full relative transition-colors cursor-pointer ${
            asPerc ? 'bg-[#FFD93D44]' : 'bg-[#1E1E1E]'
          }`}
        >
          <div className={`absolute top-0.5 w-2 h-2 rounded-full transition-all ${
            asPerc ? 'left-3.5 bg-[#FFD93D]' : 'left-0.5 bg-[#333]'
          }`} />
        </div>
        <span className="text-[9px] text-[#444]">as %</span>
      </label>

      <input
        ref={fileRef}
        type="file"
        accept=".csv"
        className="hidden"
        onChange={handleFile}
      />
    </div>
  )
}
