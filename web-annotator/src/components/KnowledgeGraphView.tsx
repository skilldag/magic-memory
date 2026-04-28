import { useEffect, useState } from 'react'
import { KnowledgeGraph } from './KnowledgeGraph'
import { DocumentViewer } from './DocumentViewer'
import { ExploreDialog } from './ExploreDialog'
import { useKnowledgeGraphStore } from '../store/knowledgeGraphStore'
import type { Concept } from '../types'

interface KnowledgeGraphViewProps {
  mode?: 'browse' | 'learn' | 'review'
}

export function KnowledgeGraphView({ mode = 'browse' }: KnowledgeGraphViewProps) {
  const [selectedConcept, setSelectedConcept] = useState<Concept | null>(null)
  const [panelSection, setPanelSection] = useState<'problem' | 'detail' | 'navigation'>('problem')
  const [showExploreDialog, setShowExploreDialog] = useState(false)

  const concepts = useKnowledgeGraphStore(state => state.concepts)
  const edges = useKnowledgeGraphStore(state => state.edges)
  const loadGraph = useKnowledgeGraphStore(state => state.loadGraph)
  const viewMode = useKnowledgeGraphStore(state => state.viewMode)
  const setViewMode = useKnowledgeGraphStore(state => state.setViewMode)

  useEffect(() => {
    loadGraph()
  }, [loadGraph])

  const handleSelectConcept = (concept: Concept) => {
    setSelectedConcept(concept)
    setPanelSection('problem')
  }

  const handleNavigate = (conceptId: string) => {
    const concept = concepts.find(c => c.id === conceptId)
    if (concept) {
      setSelectedConcept(concept)
      setPanelSection('problem')
    }
  }

  const modeButtons = [
    { key: 'browse' as const, label: '浏览' },
    { key: 'learn' as const, label: '学习' },
    { key: 'review' as const, label: '复习' },
  ]

  const tabButtons = [
    { key: 'problem' as const, label: '从问题出发' },
    { key: 'detail' as const, label: '深入理解' },
    { key: 'navigation' as const, label: '知识关联' },
  ]

  const getRelatedConcepts = (conceptId: string) => {
    return edges
      .filter(e => e.source === conceptId || e.target === conceptId)
      .map(e => {
        const otherId = e.source === conceptId ? e.target : e.source
        const concept = concepts.find(c => c.id === otherId)
        return concept ? { concept, edgeType: e.type } : null
      })
      .filter(Boolean) as { concept: Concept; edgeType: string }[]
  }

  const getDependencyChain = (conceptId: string) => {
    const chain: Concept[] = []
    let current = concepts.find(c => c.id === conceptId)
    while (current && current.depends_on.length > 0) {
      const parentId = current.depends_on[0]
      const parent = concepts.find(c => c.id === parentId)
      if (parent) {
        chain.unshift(parent)
        current = parent
      } else break
    }
    return chain
  }

  const getWhatIsSummary = (concept: Concept) => {
    const plain = concept.content
      .replace(/```[\s\S]*?```/g, '')
      .replace(/^#+\s+/gm, '')
      .replace(/[*_`>#-]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim()

    if (!plain) return ''
    return plain.length > 120 ? `${plain.slice(0, 120)}...` : plain
  }

  const getNameReason = (concept: Concept) => {
    if (concept.alias && concept.alias.length > 0) {
      return `该名称常与别名「${concept.alias.join(' / ')}」一起使用，强调同一概念在不同语境下的表达。`
    }
    if (concept.category) {
      return `名称通常用于表达其在「${concept.category}」中的职责，可结合“它解决的问题”一起理解。`
    }
    return '命名由来尚未记录，可先结合其核心问题与上下游概念理解该术语。'
  }

  const getConfusableConcepts = (concept: Concept) => {
    const pool = concepts.filter(c => c.id !== concept.id)

    const scored = pool
      .map(candidate => {
        let score = 0
        if (candidate.category === concept.category) score += 2
        if (Math.abs(candidate.level - concept.level) <= 1) score += 1
        if (concept.related.includes(candidate.id) || candidate.related.includes(concept.id)) score += 2
        if (concept.depends_on.includes(candidate.id) || concept.leads_to.includes(candidate.id)) score += 1
        if (candidate.depends_on.includes(concept.id) || candidate.leads_to.includes(concept.id)) score += 1
        return { candidate, score }
      })
      .filter(item => item.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 3)

    return scored.map(item => item.candidate)
  }

  const getDiffHint = (base: Concept, other: Concept) => {
    if (base.category !== other.category) {
      return `「${base.title}」偏向 ${base.category}，而「${other.title}」更偏向 ${other.category}。`
    }
    if (base.level !== other.level) {
      return `两者都在 ${base.category}，但「${base.title}」是 L${base.level}，而「${other.title}」是 L${other.level}。`
    }
    if (other.problem) {
      return `对比时先看它解决的问题：${other.problem}`
    }
    return '两者在知识图谱中位置接近，建议对比“解决的问题”和“前置概念”来区分。'
  }

  const getRelationReason = (current: Concept, other: Concept, edgeType: string) => {
    if (edgeType === 'depends_on') {
      if (other.problem) return `先理解它：${other.problem}`
      return `它为「${current.title}」提供前置认知基础。`
    }
    if (edgeType === 'leads_to') {
      if (other.problem) return `学完当前概念后，下一步常会遇到：${other.problem}`
      return `它是「${current.title}」的自然延伸方向。`
    }
    if (current.category !== other.category) {
      return `它从「${other.category}」视角补充了当前概念。`
    }
    return '它与当前概念在同一主题中互补，可并行理解。'
  }

  return (
    <div className="flex h-full w-full overflow-hidden">
      {/* 左侧图谱 - 占用剩余空间 */}
      <div className="flex-1 min-w-0 relative">
        <div className="absolute top-3 left-3 z-10 flex items-center gap-1 bg-white/90 backdrop-blur rounded-lg shadow px-1.5 py-1">
          {modeButtons.map(b => (
            <button
              key={b.key}
              onClick={() => setViewMode(b.key)}
              className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                viewMode === b.key
                  ? 'bg-blue-500 text-white shadow-sm'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              {b.label}
            </button>
          ))}
        </div>
        <KnowledgeGraph
          concepts={concepts}
          edges={edges}
          selectedConcept={selectedConcept}
          onSelectConcept={handleSelectConcept}
          onNavigate={handleNavigate}
        />
      </div>

      {/* 右侧内容面板 - 固定宽度，始终可见 */}
      <div className="w-[420px] xl:w-[480px] shrink-0 border-l border-gray-200 bg-white flex flex-col overflow-hidden">
        {selectedConcept ? (
          <>
            {/* 头部 */}
            <div className="shrink-0 px-5 pt-4 pb-3 border-b border-gray-100">
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <h2 className="text-base font-semibold text-gray-900 truncate">
                    {selectedConcept.title}
                  </h2>
                  {selectedConcept.alias && selectedConcept.alias.length > 0 && (
                    <p className="text-xs text-gray-500 mt-0.5 truncate">
                      别名: {selectedConcept.alias.join(' / ')}
                    </p>
                  )}
                </div>
                <button
                  onClick={() => setSelectedConcept(null)}
                  className="shrink-0 ml-2 p-1 rounded hover:bg-gray-100 text-gray-400 hover:text-gray-600"
                  title="取消选择"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              <div className="flex items-center gap-2 mt-2">
                <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                  selectedConcept.level === 1 ? 'bg-green-100 text-green-700' :
                  selectedConcept.level === 2 ? 'bg-blue-100 text-blue-700' :
                  'bg-purple-100 text-purple-700'
                }`}>
                  L{selectedConcept.level}
                </span>
                <span className="px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-600">
                  {selectedConcept.category}
                </span>
              </div>
            </div>

            {/* Tab 切换 */}
            <div className="shrink-0 flex border-b border-gray-100">
              {tabButtons.map(t => (
                <button
                  key={t.key}
                  onClick={() => setPanelSection(t.key)}
                  className={`flex-1 px-3 py-2.5 text-sm font-medium text-center transition-colors ${
                    panelSection === t.key
                      ? 'text-blue-600 border-b-2 border-blue-500'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  {t.label}
                </button>
              ))}
            </div>

            {/* 内容区域 */}
            <div className="flex-1 overflow-y-auto">
              {/* 从问题出发 */}
              {panelSection === 'problem' && (
                <div className="px-5 py-4 space-y-4">
                  <div>
                    <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                      这个概念是什么
                    </h3>
                    <div className="p-3 bg-indigo-50 border-l-4 border-indigo-400 rounded-r-lg">
                      <p className="text-sm leading-relaxed text-indigo-900">
                        {getWhatIsSummary(selectedConcept) || '定义尚未补充，可先查看“深入理解”获取完整说明。'}
                      </p>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                      它所解决的问题
                    </h3>
                    <div className="p-3 bg-amber-50 border-l-4 border-amber-400 rounded-r-lg">
                      <p className="text-sm leading-relaxed text-amber-900 font-medium">
                        {selectedConcept.problem || '背景问题尚未记录'}
                      </p>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                      为什么叫这个名字
                    </h3>
                    <div className="p-3 bg-violet-50 border-l-4 border-violet-400 rounded-r-lg">
                      <p className="text-sm leading-relaxed text-violet-900">
                        {getNameReason(selectedConcept)}
                      </p>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                      容易混淆的概念
                    </h3>
                    <div className="space-y-1.5">
                      {(() => {
                        const confusable = getConfusableConcepts(selectedConcept)
                        if (confusable.length === 0) {
                          return (
                            <div className="px-3 py-2 rounded-lg border border-gray-200 bg-gray-50">
                              <p className="text-xs leading-relaxed text-gray-600">
                                暂无可对比概念，后续可补充“同类概念”后进行区分学习。
                              </p>
                            </div>
                          )
                        }

                        return confusable.map(concept => (
                          <button
                            key={concept.id}
                            onClick={() => handleNavigate(concept.id)}
                            className="w-full text-left px-3 py-2 rounded-lg border border-purple-200 hover:border-purple-300 hover:bg-purple-50 transition-colors group"
                          >
                            <div className="flex items-center justify-between">
                              <span className="text-sm font-medium text-gray-800 group-hover:text-purple-700">
                                {concept.title}
                              </span>
                              <span className="text-xs text-purple-300 group-hover:text-purple-500">对比</span>
                            </div>
                            <p className="text-xs text-gray-500 mt-0.5">{getDiffHint(selectedConcept, concept)}</p>
                          </button>
                        ))
                      })()}
                    </div>
                  </div>

                  {selectedConcept.depends_on.length > 0 && (
                    <div>
                      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                        先理解这些前置概念
                      </h3>
                      <div className="space-y-1.5">
                        {selectedConcept.depends_on.map(id => {
                          const parent = concepts.find(c => c.id === id)
                          if (!parent) return null
                          return (
                            <button
                              key={id}
                              onClick={() => handleNavigate(parent.id)}
                              className="w-full text-left px-3 py-2 rounded-lg border border-gray-200 hover:border-blue-300 hover:bg-blue-50 transition-colors group"
                            >
                              <div className="flex items-center justify-between">
                                <span className="text-sm font-medium text-gray-800 group-hover:text-blue-700">
                                  {parent.title}
                                </span>
                                <span
                                  className="shrink-0 text-[14px] leading-none text-gray-500 group-hover:text-blue-500"
                                  aria-hidden
                                >
                                  ›
                                </span>
                              </div>
                              {parent.problem && (
                                <p className="text-xs text-gray-500 mt-0.5">{parent.problem}</p>
                              )}
                            </button>
                          )
                        })}
                      </div>
                    </div>
                  )}

                  <div>
                    <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                      你可能产生的疑问
                    </h3>
                    <div className="p-3 bg-sky-50 border-l-4 border-sky-400 rounded-r-lg">
                      <p className="text-sm leading-relaxed text-sky-900">
                        {selectedConcept.gap_anticipate || '暂无预期疑问'}
                      </p>
                    </div>
                  </div>

                  {/^\d+$/.test(selectedConcept.id) && (
                    <div>
                      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                        数字锚点
                      </h3>
                      <div className="flex items-center gap-2 p-3 bg-gray-50 rounded-lg border border-gray-200">
                        <span className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-400 to-purple-500 flex items-center justify-center text-white text-sm font-bold">
                          {selectedConcept.id}
                        </span>
                        <span className="text-sm text-gray-600">
                          数字 <span className="font-medium text-gray-800">{selectedConcept.id}</span>
                          {selectedConcept.alias && selectedConcept.alias.length > 0 && (
                            <> — 锚点: {selectedConcept.alias[0]}</>
                          )}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* 深入理解 */}
              {panelSection === 'detail' && (
                <div className="px-5 py-4">
                  <div className="prose prose-sm max-w-none">
                    <DocumentViewer document={{
                      id: selectedConcept.id,
                      title: selectedConcept.title,
                      path: selectedConcept.path,
                      content: selectedConcept.content,
                      level: selectedConcept.level,
                      category: selectedConcept.category,
                      tags: selectedConcept.tags,
                      lastModified: selectedConcept.lastModified,
                      metadata: selectedConcept.metadata
                    }} />
                  </div>
                </div>
              )}

              {/* 知识关联 */}
              {panelSection === 'navigation' && (
                <div className="px-5 py-4 space-y-5">
                  <div>
                    <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                      推导路径
                    </h3>
                    <div className="p-4 bg-white rounded-lg border border-gray-200">
                      <svg viewBox="0 0 400 120" className="w-full h-auto">
                        {(() => {
                          const chain = getDependencyChain(selectedConcept.id)
                          const allNodes = [...chain, selectedConcept]
                          const nodeW = 80, gap = 30
                          const totalW = allNodes.length * nodeW + (allNodes.length - 1) * gap
                          const startX = Math.max(10, (400 - totalW) / 2)
                          const y = 60
                          return (
                            <g>
                              {allNodes.map((node, i) => {
                                const x = startX + i * (nodeW + gap)
                                const isCurrent = node.id === selectedConcept.id
                                return (
                                  <g key={node.id}>
                                    {i > 0 && (
                                      <line x1={startX + (i-1)*(nodeW+gap) + nodeW} y1={y} x2={x} y2={y} stroke="#d1d5db" strokeWidth={1.5} markerEnd="url(#arrow)" />
                                    )}
                                    <rect x={x} y={y-18} width={nodeW} height={36} rx={8} fill={isCurrent ? '#3b82f6' : '#f3f4f6'} stroke={isCurrent ? '#2563eb' : '#e5e7eb'} strokeWidth={1} />
                                    <text x={x+nodeW/2} y={y+4} textAnchor="middle" fontSize={10} fill={isCurrent ? 'white' : '#374151'}>{node.title.length > 8 ? node.title.slice(0,7)+'…' : node.title}</text>
                                  </g>
                                )
                              })}
                              <defs><marker id="arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="#d1d5db"/></marker></defs>
                            </g>
                          )
                        })()}
                      </svg>
                    </div>
                  </div>

                  {(() => {
                    const related = getRelatedConcepts(selectedConcept.id)
                    const byType: Record<string, {concept: Concept; edgeType: string}[]> = {
                      leads_to: related.filter(r => r.edgeType === 'leads_to'),
                      depends_on: related.filter(r => r.edgeType === 'depends_on'),
                      related: related.filter(r => r.edgeType === 'related'),
                    }
                    return (
                      <>
                        {byType.leads_to.length > 0 && (
                          <div>
                            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">由此推出</h3>
                            <div className="space-y-1.5">
                              {byType.leads_to.map(({ concept }) => (
                                <button key={concept.id} onClick={() => handleNavigate(concept.id)} className="w-full text-left px-3 py-2 rounded-lg border border-green-200 hover:border-green-300 hover:bg-green-50 transition-colors group">
                                  <div className="flex items-center justify-between">
                                    <span className="text-sm font-medium text-gray-800 group-hover:text-green-700">{concept.title}</span>
                                    <span className="text-xs text-gray-400">→</span>
                                  </div>
                                  <p className="text-xs text-gray-500 mt-0.5">{getRelationReason(selectedConcept, concept, 'leads_to')}</p>
                                </button>
                              ))}
                            </div>
                          </div>
                        )}
                        {byType.depends_on.length > 0 && (
                          <div>
                            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">前置基础</h3>
                            <div className="space-y-1.5">
                              {byType.depends_on.map(({ concept }) => (
                                <button key={concept.id} onClick={() => handleNavigate(concept.id)} className="w-full text-left px-3 py-2 rounded-lg border border-red-200 hover:border-red-300 hover:bg-red-50 transition-colors group">
                                  <div className="flex items-center justify-between">
                                    <span className="text-sm font-medium text-gray-800 group-hover:text-red-700">{concept.title}</span>
                                    <span className="text-xs text-gray-400">←</span>
                                  </div>
                                  <p className="text-xs text-gray-500 mt-0.5">{getRelationReason(selectedConcept, concept, 'depends_on')}</p>
                                </button>
                              ))}
                            </div>
                          </div>
                        )}
                        {byType.related.length > 0 && (
                          <div>
                            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">平行关联</h3>
                            <div className="space-y-1.5">
                              {byType.related.map(({ concept }) => (
                                <button key={concept.id} onClick={() => handleNavigate(concept.id)} className="w-full text-left px-3 py-2 rounded-lg border border-gray-200 hover:border-gray-300 hover:bg-gray-50 transition-colors group">
                                  <div className="flex items-center justify-between">
                                    <span className="text-sm font-medium text-gray-800 group-hover:text-gray-900">{concept.title}</span>
                                    <span className="text-xs text-gray-400">↔</span>
                                  </div>
                                  <p className="text-xs text-gray-500 mt-0.5">{getRelationReason(selectedConcept, concept, 'related')}</p>
                                </button>
                              ))}
                            </div>
                          </div>
                        )}
                        {related.length === 0 && (
                          <div className="flex flex-col items-center justify-center py-10 text-gray-400">
                            <svg className="w-10 h-10 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"/></svg>
                            <p className="text-sm">暂未记录关联关系</p>
                          </div>
                        )}
                      </>
                    )
                  })()}

                  {/* 探索入口 */}
                  <div className="border-t border-gray-100 pt-4 mt-2">
                    <button
                      onClick={() => setShowExploreDialog(true)}
                      className="w-full flex items-center gap-3 px-4 py-3 rounded-lg border-2 border-dashed border-blue-300 text-blue-600 hover:bg-blue-50 hover:border-blue-400 transition-colors group"
                    >
                      <svg className="w-5 h-5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                      </svg>
                      <div className="text-left">
                        <div className="text-sm font-medium group-hover:text-blue-700">从此概念继续探索</div>
                        <div className="text-xs text-blue-400 group-hover:text-blue-500">添加新的关联概念到知识图谱</div>
                      </div>
                    </button>
                  </div>
                </div>
              )}
            </div>
          </>
        ) : (
          /* 空状态 - 未选中任何概念 */
          <div className="flex-1 flex flex-col items-center justify-center text-gray-400 px-8">
            <svg className="w-16 h-16 mb-4 text-gray-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
            <p className="text-sm text-center leading-relaxed">点击图谱中的节点<br/>查看概念详情</p>
          </div>
        )}
      </div>
      {showExploreDialog && selectedConcept && (
        <ExploreDialog
          sourceConcept={selectedConcept}
          onClose={() => setShowExploreDialog(false)}
        />
      )}
    </div>
  )
}
