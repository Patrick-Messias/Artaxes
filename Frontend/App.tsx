import React, { useState } from 'react'
import { Sidebar }      from './components/Sidebar/Sidebar'
import { Basket }       from './components/Basket/Basket'
import { EquityChart }  from './components/Chart/EquityChart'

export default function App() {
  const [fontSize, setFontSize] = useState(11)

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#0D0D0D] text-white font-mono">

      {/* ── Sidebar ───────────────────────────────────────────────────── */}
      <Sidebar fontSize={fontSize} />

      {/* ── Main ─────────────────────────────────────────────────────── */}
      <div className="flex flex-col flex-1 overflow-hidden min-w-0">

        {/* Top bar */}
        <div className="flex items-center gap-4 px-4 py-2 border-b border-[#1E1E1E] shrink-0">
          <span className="text-[#00D9FF] tracking-[0.2em] text-[10px] font-semibold">
            ART BACKTESTING
          </span>

          <div className="ml-auto flex items-center gap-4">
            {/* Font size slider */}
            <label className="flex items-center gap-2 text-[10px] text-[#444]">
              <span>font</span>
              <input
                type="range" min={9} max={14} value={fontSize}
                onChange={e => setFontSize(Number(e.target.value))}
                className="w-16 accent-[#00D9FF]"
              />
              <span className="w-5 text-[#666]">{fontSize}</span>
            </label>
          </div>
        </div>

        {/* Content — chart + basket side by side */}
        <div className="flex flex-1 overflow-hidden min-h-0">
          <div className="flex-1 overflow-auto p-4 min-w-0">
            <EquityChart />
          </div>
          <Basket />
        </div>

      </div>
    </div>
  )
}
