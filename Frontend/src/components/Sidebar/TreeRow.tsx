import React from 'react'
import type { ParamSet, FilterMode } from '../../types'
import { useBasket } from '../../store/basket'

// ── Generic expandable tree row ───────────────────────────────────────────
interface RowProps {
  depth:    number
  label:    string
  meta?:    string
  expanded?: boolean
  onClick:  () => void
}

export function Row({ depth, label, meta, expanded, onClick }: RowProps) {
  return (
    <button
      onClick={onClick}
      className="w-full flex items-center gap-1 py-[3px] hover:bg-[#111] text-left transition-colors"
      style={{ paddingLeft: 8 + depth * 12 }}
    >
      <span className="text-[#383838] text-[9px] w-3 shrink-0">
        {expanded !== undefined ? (expanded ? '▾' : '▸') : '·'}
      </span>
      <span className="text-[#AAAAAA] truncate flex-1 leading-tight">{label}</span>
      {meta && (
        <span className="text-[#383838] text-[9px] shrink-0 pr-2">{meta}</span>
      )}
    </button>
  )
}

// ── Leaf node — param_set with add buttons ────────────────────────────────
interface ParamSetRowProps {
  ps:      ParamSet
  depth:   number
  filters: Set<FilterMode>
}

// Em TreeRow.tsx — Nova versão do ParamSetRow
export function ParamSetRow({ ps, depth, filters }: ParamSetRowProps) {
  const { toggle, has } = useBasket()

  // Fallback para nomes ou PnL nulos vindos da API
  const displayName = (ps as any).ps_name || ps.name || "Unnamed Parset";
  const pnl = ps.total_pnl ?? 0
  const pnlLabel = ps.total_pnl != null ? `${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}%` : "---"

  return (
    <div
      className="flex items-center py-[2px] hover:bg-[#0F0F0F] group transition-colors"
      style={{ paddingLeft: 8 + depth * 12 }}
    >
      <span className="text-[#2A2A2A] text-[9px] w-3 shrink-0">·</span>

      <span className="text-[#888] truncate flex-1 text-[10px] leading-tight">
        {displayName}
      </span>

      {/* PnL visível por padrão */}
      <span className={`text-[9px] shrink-0 mr-1 group-hover:hidden ${pnl >= 0 ? 'text-[#4ECB71]' : 'text-[#FF6B6B]'}`}>
        {pnlLabel}
      </span>

      {/* Botões de Ação */}
      <div className="hidden group-hover:flex gap-1 pr-1">
        {filters.has('parset') && (
          <button
            onClick={() => toggle(ps, 'parset')}
            className={`text-[8px] px-1.5 py-0.5 rounded transition-all ${
              has(ps.id, 'parset') ? 'bg-[#00D9FF] text-black font-bold' : 'bg-[#1A1A1A] text-[#555]'
            }`}
          >
            PS
          </button>
        )}
        
        {/* O BOTÃO SÓ APARECE SE wf_json_path NÃO FOR NULL */}
        {filters.has('wf') && ps.wf_json_path && (
          <button
            onClick={() => toggle(ps, 'wf')}
            className={`text-[8px] px-1.5 py-0.5 rounded transition-all ${
              has(ps.id, 'wf') ? 'bg-[#C77DFF] text-black font-bold' : 'bg-[#1A1A1A] text-[#555]'
            }`}
          >
            WF
          </button>
        )}
      </div>
    </div>
  )
}

// export function ParamSetRow({ ps, depth, filters }: ParamSetRowProps) {
//   const { toggle, has } = useBasket()

//   const pnl      = ps.total_pnl
//   const pnlLabel = pnl != null
//     ? `${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}%`
//     : null
//   const pnlColor = (pnl ?? 0) >= 0 ? 'text-[#4ECB71]' : 'text-[#FF6B6B]'

//   return (
//     <div
//       className="flex items-center py-[2px] hover:bg-[#0F0F0F] group transition-colors"
//       style={{ paddingLeft: 8 + depth * 12 }}
//     >
//       <span className="text-[#2A2A2A] text-[9px] w-3 shrink-0">·</span>

//       {/* Name */}
//       <span className="text-[#888] truncate flex-1 text-[10px] leading-tight">
//         {ps.ps_name}
//       </span>

//       {/* PnL — hidden on hover to show buttons */}
//       {pnlLabel && (
//         <span className={`text-[9px] shrink-0 mr-1 group-hover:hidden ${pnlColor}`}>
//           {pnlLabel}
//         </span>
//       )}

//       {/* Add buttons — shown on hover */}
//       <div className="hidden group-hover:flex gap-1 pr-1">
//         {filters.has('parset') && (
//           <AddBtn
//             label="PS"
//             color="#00D9FF"
//             active={has(ps.id, 'parset')}
//             onClick={() => toggle(ps, 'parset')}
//           />
//         )}
//         {filters.has('wf') && ps.wf_json_path && (
//           <AddBtn
//             label="WF"
//             color="#C77DFF"
//             active={has(ps.id, 'wf')}
//             onClick={() => toggle(ps, 'wf')}
//           />
//         )}
//         {filters.has('selected') && ps.best_ps_name && (
//           <AddBtn
//             label="SEL"
//             color="#4ECB71"
//             active={has(ps.id, 'selected')}
//             onClick={() => toggle(ps, 'selected')}
//           />
//         )}
//       </div>
//     </div>
//   )
// }

// ── Add button ────────────────────────────────────────────────────────────
interface AddBtnProps {
  label:   string
  color:   string
  active:  boolean
  onClick: () => void
}

function AddBtn({ label, color, active, onClick }: AddBtnProps) {
  return (
    <button
      onClick={e => { e.stopPropagation(); onClick() }}
      className="text-[8px] px-1.5 py-0.5 rounded transition-all"
      style={active
        ? { backgroundColor: color, color: '#000', fontWeight: 600 }
        : { backgroundColor: '#1A1A1A', color: '#555' }
      }
    >
      {label}
    </button>
  )
}
