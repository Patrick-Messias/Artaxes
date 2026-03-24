import React from 'react'
import { useBasket } from '../../store/basket'
import type { BasketItem } from '../../types'

interface Props {
  item: BasketItem
}

const TYPE_STYLE: Record<BasketItem['type'], string> = {
  parset:   'text-[#00D9FF] bg-[#00D9FF11]',
  wf:       'text-[#C77DFF] bg-[#C77DFF11]',
  selected: 'text-[#4ECB71] bg-[#4ECB7111]',
}

export function BasketItemComp({ item }: Props) {
  const { remove } = useBasket()

  const pnl      = item.ps.total_pnl
  const pnlLabel = pnl != null
    ? `${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}%`
    : null
  const pnlColor = (pnl ?? 0) >= 0 ? 'text-[#4ECB71]' : 'text-[#FF6B6B]'

  return (
    <div className="flex items-start gap-2 px-3 py-2 border-b border-[#0F0F0F] group hover:bg-[#0F0F0F] transition-colors">

      {/* Color dot */}
      <div
        className="w-1.5 h-1.5 rounded-full mt-1.5 shrink-0"
        style={{ backgroundColor: item.color }}
      />

      <div className="flex-1 min-w-0">
        {/* Type badge */}
        <span className={`text-[8px] px-1.5 py-0.5 rounded ${TYPE_STYLE[item.type]}`}>
          {item.type.toUpperCase()}
        </span>

        {/* Label */}
        <div className="text-[10px] text-[#888] truncate mt-1 leading-tight">
          {item.label}
        </div>

        {/* Metrics */}
        <div className="flex gap-2 mt-1">
          {item.ps.n_trades != null && (
            <span className="text-[9px] text-[#444]">
              {item.ps.n_trades}t
            </span>
          )}
          {pnlLabel && (
            <span className={`text-[9px] ${pnlColor}`}>
              {pnlLabel}
            </span>
          )}
          {item.ps.exec_tf && (
            <span className="text-[9px] text-[#333]">
              {item.ps.exec_tf}
            </span>
          )}
        </div>
      </div>

      {/* Remove button */}
      <button
        onClick={() => remove(item.id)}
        className="text-[#2A2A2A] hover:text-[#FF6B6B] text-[14px] leading-none opacity-0 group-hover:opacity-100 transition-all mt-0.5 shrink-0"
      >
        ×
      </button>
    </div>
  )
}
