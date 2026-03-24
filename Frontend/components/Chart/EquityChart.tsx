import React, { useState, useEffect, useCallback } from 'react'
import Plot                  from 'react-plotly.js'
import { useBasket }         from '../../store/basket'
import { api }               from '../../api/client'
import { BenchmarkSelector } from './BenchmarkSelector'
import type { EquityPoint }  from '../../types'

// ── Plotly layout base — dark theme ──────────────────────────────────────
const BASE_LAYOUT: Partial<Plotly.Layout> = {
  paper_bgcolor: 'transparent',
  plot_bgcolor:  'transparent',
  font:  { color: '#555', size: 10, family: 'JetBrains Mono, monospace' },
  xaxis: {
    gridcolor:  '#161616',
    linecolor:  '#1E1E1E',
    tickcolor:  '#2A2A2A',
    zeroline:   false,
    tickfont:   { size: 9, color: '#444' },
  },
  yaxis: {
    gridcolor:   '#161616',
    linecolor:   '#1E1E1E',
    tickcolor:   '#2A2A2A',
    zeroline:    true,
    zerolinecolor: '#2A2A2A',
    ticksuffix:  '%',
    tickfont:    { size: 9, color: '#444' },
  },
  legend: {
    bgcolor:     '#0A0A0A',
    bordercolor: '#1E1E1E',
    borderwidth: 1,
    font:        { size: 9, color: '#555' },
    x: 0, y: 1,
    xanchor: 'left',
    yanchor: 'top',
  },
  margin:    { t: 8, r: 12, b: 36, l: 52 },
  hovermode: 'x unified',
  hoverlabel: {
    bgcolor:     '#0D0D0D',
    bordercolor: '#2A2A2A',
    font:        { size: 9, color: '#AAA', family: 'JetBrains Mono, monospace' },
  },
}

export function EquityChart() {
  const { items } = useBasket()

  const [groupEc,      setGroupEc]      = useState(false)
  const [traces,       setTraces]       = useState<Plotly.Data[]>([])
  const [loading,      setLoading]      = useState(false)
  const [error,        setError]        = useState<string | null>(null)
  const [benchmark,    setBenchmark]    = useState<EquityPoint[] | null>(null)
  const [benchmarkPct, setBenchmarkPct] = useState(true)

  // Fetch equity curve when basket or groupEc changes
  useEffect(() => {
    if (items.length === 0) {
      setTraces([])
      setError(null)
      return
    }

    setLoading(true)
    setError(null)

    const ps_ids = [...new Set(items.map(i => i.ps_id))]

    api.getEquityCurve(ps_ids)
      .then(points => {
        if (groupEc) {
          // Single aggregated curve — sum of all
          setTraces([{
            x:    points.map(p => p.datetime),
            y:    points.map(p => p.cumulative_pnl),
            type: 'scatter',
            mode: 'lines',
            name: `Portfolio (${items.length})`,
            line: { color: '#00D9FF', width: 2 },
          }])
        } else {
          // Individual curve per basket item
          const t: Plotly.Data[] = items.map(item => ({
            x:    points.map(p => p.datetime),
            y:    points.map(p => (p[`ps_${item.ps_id}`] as number) ?? null),
            type: 'scatter',
            mode: 'lines',
            name: item.label,
            line: { color: item.color, width: 1.5 },
            connectgaps: false,
          }))
          setTraces(t)
        }
        setLoading(false)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
      })
  }, [items, groupEc])

  // Add benchmark trace
  const allTraces: Plotly.Data[] = [...traces]
  if (benchmark) {
    allTraces.push({
      x:    benchmark.map(p => p.datetime),
      y:    benchmark.map(p => benchmarkPct ? p.cumulative_pnl : p.total_pnl),
      type: 'scatter',
      mode: 'lines',
      name: 'Benchmark',
      line: { color: '#FFD93D', width: 1.5, dash: 'dash' },
    })
  }

  return (
    <div className="flex flex-col h-full gap-3 min-h-0">

      {/* ── Controls ──────────────────────────────────────────────── */}
      <div className="flex items-center gap-4 shrink-0">
        <span className="text-[10px] tracking-widest text-[#333]">
          EQUITY CURVE
        </span>

        {/* Group toggle */}
        <label className="flex items-center gap-2 cursor-pointer">
          <div
            onClick={() => setGroupEc(g => !g)}
            className={`w-8 h-4 rounded-full relative transition-colors cursor-pointer ${
              groupEc ? 'bg-[#00D9FF33]' : 'bg-[#1A1A1A]'
            }`}
          >
            <div className={`absolute top-0.5 w-3 h-3 rounded-full transition-all ${
              groupEc ? 'left-4 bg-[#00D9FF]' : 'left-0.5 bg-[#333]'
            }`} />
          </div>
          <span className="text-[10px] text-[#444]">group</span>
        </label>

        {/* Benchmark */}
        <BenchmarkSelector
          onSelect={setBenchmark}
          onClear={() => setBenchmark(null)}
          hasBenchmark={benchmark !== null}
          asPerc={benchmarkPct}
          onTogglePerc={() => setBenchmarkPct(p => !p)}
        />

        {/* Status */}
        <div className="ml-auto">
          {loading && (
            <span className="text-[10px] text-[#00D9FF] animate-pulse">
              loading...
            </span>
          )}
          {error && (
            <span className="text-[10px] text-[#FF6B6B]" title={error}>
              error
            </span>
          )}
          {!loading && !error && items.length === 0 && (
            <span className="text-[10px] text-[#2A2A2A] italic">
              add items to basket
            </span>
          )}
          {!loading && !error && items.length > 0 && (
            <span className="text-[10px] text-[#333]">
              {items.length} series
            </span>
          )}
        </div>
      </div>

      {/* ── Plot ──────────────────────────────────────────────────── */}
      <div className="flex-1 min-h-0">
        {allTraces.length > 0 ? (
          <Plot
            data={allTraces}
            layout={BASE_LAYOUT as Plotly.Layout}
            config={{
              displayModeBar:  false,
              responsive:      true,
              scrollZoom:      true,
            }}
            style={{ width: '100%', height: '100%' }}
            useResizeHandler
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <span className="text-[#1E1E1E] text-xs tracking-widest">
              NO DATA
            </span>
          </div>
        )}
      </div>

    </div>
  )
}
