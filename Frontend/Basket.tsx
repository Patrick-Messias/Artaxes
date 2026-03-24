import React from 'react'
import { useBasket }       from '../../store/basket'
import { BasketItemComp }  from './BasketItem'

export function Basket() {
  const { items, clear } = useBasket()

  const totalPnl = items.reduce((sum, i) => sum + (i.ps.total_pnl ?? 0), 0)
  const hasPnl   = items.some(i => i.ps.total_pnl != null)

  return (
    <div className="w-52 border-l border-[#1E1E1E] bg-[#0A0A0A] flex flex-col shrink-0 overflow-hidden">

      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-[#1E1E1E] shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-[10px] tracking-widest text-[#444]">BASKET</span>
          {items.length > 0 && (
            <span className="text-[9px] text-[#333] bg-[#1A1A1A] px-1.5 py-0.5 rounded">
              {items.length}
            </span>
          )}
        </div>
        {items.length > 0 && (
          <button
            onClick={clear}
            className="text-[9px] text-[#333] hover:text-[#FF6B6B] transition-colors"
          >
            clear
          </button>
        )}
      </div>

      {/* Items */}
      <div className="flex-1 overflow-y-auto">
        {items.length === 0 ? (
          <div className="p-4 text-[10px] text-[#2A2A2A] italic leading-relaxed">
            select items from<br />sidebar to plot
          </div>
        ) : (
          items.map(item => (
            <BasketItemComp key={item.id} item={item} />
          ))
        )}
      </div>

      {/* Footer — total PnL summary */}
      {items.length > 0 && hasPnl && (
        <div className="px-3 py-2 border-t border-[#1E1E1E] shrink-0">
          <div className="flex justify-between items-center">
            <span className="text-[9px] text-[#333]">avg pnl</span>
            <span className={`text-[10px] ${totalPnl / items.length >= 0 ? 'text-[#4ECB71]' : 'text-[#FF6B6B]'}`}>
              {(totalPnl / items.length) >= 0 ? '+' : ''}
              {(totalPnl / items.length).toFixed(2)}%
            </span>
          </div>
        </div>
      )}

    </div>
  )
}
