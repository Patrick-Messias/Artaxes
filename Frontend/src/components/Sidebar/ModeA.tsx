import React, { useState } from 'react'
import { Row, ParamSetRow } from './TreeRow'
import type { Operation, ParamSet, FilterMode } from '../../types'

interface Props {
  operations: Operation[]
  paramSets:  ParamSet[]
  filters:    Set<FilterMode>
}

export function ModeA({ operations, paramSets, filters }: Props) {
  const [open, setOpen] = useState<Set<string>>(new Set())

  const toggle = (key: string) => setOpen(prev => {
    const next = new Set(prev)
    next.has(key) ? next.delete(key) : next.add(key)
    return next
  })

  return (
    <div>
      {operations.map(op => {
        const opPs   = paramSets.filter(p => p.operation_name === op.name)
        const models = [...new Set(opPs.map(p => p.model_name))].sort()

        return (
          <div key={op.id}>
            {/* Operation */}
            <Row
              depth={0}
              label={op.name}
              meta={`${op.n_param_sets}ps`}
              expanded={open.has(op.name)}
              onClick={() => toggle(op.name)}
            />

            {open.has(op.name) && models.map(model => {
              const modelPs = opPs.filter(p => p.model_name === model)
              const strats  = [...new Set(modelPs.map(p => p.strat_name))].sort()
              const mKey    = `${op.name}/${model}`

              return (
                <div key={model}>
                  {/* Model */}
                  <Row
                    depth={1}
                    label={model}
                    meta={`${modelPs.length}ps`}
                    expanded={open.has(mKey)}
                    onClick={() => toggle(mKey)}
                  />

                  {open.has(mKey) && strats.map(strat => {
                    const stratPs = modelPs.filter(p => p.strat_name === strat)
                    const assets  = [...new Set(stratPs.map(p => p.asset_name))].sort()
                    const sKey    = `${mKey}/${strat}`

                    return (
                      <div key={strat}>
                        {/* Strat */}
                        <Row
                          depth={2}
                          label={strat}
                          meta={`${stratPs.length}ps`}
                          expanded={open.has(sKey)}
                          onClick={() => toggle(sKey)}
                        />

                        {open.has(sKey) && assets.map(asset => {
                          const assetPs = stratPs.filter(p => p.asset_name === asset)
                          const aKey    = `${sKey}/${asset}`

                          return (
                            <div key={asset}>
                              {/* Asset */}
                              <Row
                                depth={3}
                                label={asset}
                                meta={`${assetPs.length}ps`}
                                expanded={open.has(aKey)}
                                onClick={() => toggle(aKey)}
                              />

                              {/* Param sets */}
                              {open.has(aKey) && assetPs.map(ps => (
                                <ParamSetRow
                                  key={ps.id}
                                  ps={ps}
                                  depth={4}
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
            })}
          </div>
        )
      })}
    </div>
  )
}
