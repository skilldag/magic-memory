import { useState, useMemo, useCallback } from 'react'
import { DocumentViewer } from './DocumentViewer'

import { ComparisonPanel } from './ComparisonPanel'
import { DependencyChainSVG } from './DependencyChainSVG'
import {
  getRelatedConcepts,
  getDependencyChain,
  getRelationReason,
  getReviewRecordFor,
} from '../utils/knowledgeGraph'
import { generateReferenceFlow, diffFlows, getGapConceptIds, generateGenericChain } from '../utils/processComparison'
import { useKnowledgeGraphStore } from '../store/knowledgeGraphStore'
import type { Concept, ConceptEdge, ReviewRecord, ProcessChain } from '../types'

interface ConceptDetailPanelProps {
  concept: Concept
  concepts: Concept[]
  edges: ConceptEdge[]
  reviewRecords: Map<string, ReviewRecord>
  onNavigate: (conceptId: string) => void
  onDeselect: () => void
  onEnterProcess?: (concept: Concept) => void
}

type ActionKey = 'process' | 'compare' | 'explore' | 'read'

export function ConceptDetailPanel({
  concept,
  concepts,
  edges,
  reviewRecords,
  onNavigate,
  onDeselect,
  onEnterProcess,
}: ConceptDetailPanelProps) {
  const [action, setAction] = useState<ActionKey>('process')
  const chains = useKnowledgeGraphStore(s => s.chains)
  const updateProcessState = useKnowledgeGraphStore(s => s.updateProcessState)

  const processState = reviewRecords.get(concept.id)?.process_state

  const chain = useMemo(() => {
    if (concept.process) {
      return chains.find(ch => ch.id === concept.process.chain_id) ?? null
    }
    return generateGenericChain(concept.id, concepts)
  }, [concept.process, chains, concept.id, concepts])

  const referenceSteps = useMemo(() => {
    if (concept.process) {
      return generateReferenceFlow(concept.id, concepts, chains)
    }
    return generateGenericChain(concept.id, concepts).steps
  }, [concept.id, concepts, chains, concept.process])

  const diffs = useMemo(() => {
    if (!processState?.filled || !processState.user_flow) return []
    return diffFlows(processState.user_flow, referenceSteps)
  }, [processState?.filled, processState?.user_flow, referenceSteps])

  const handleProcessComplete = useCallback((userFlow: string[]) => {
    const llmFlow = referenceSteps.map(s => s.id)
    const diffsResult = diffFlows(userFlow, referenceSteps)
    const gaps = getGapConceptIds(diffsResult)
    updateProcessState(concept.id, {
      user_flow: userFlow,
      llm_flow: llmFlow,
      gaps,
      filled: true,
      compared: true,
    })
    setAction('compare')
  }, [concept.id, referenceSteps, updateProcessState])

  const handleNavigateGap = useCallback((conceptId: string) => {
    onNavigate(conceptId)
  }, [onNavigate])

  const actions: { key: ActionKey; label: string; desc: string }[] = [
    { key: 'process', label: '梳理过程', desc: '推导流程中的步骤' },
    { key: 'compare', label: '对照验证', desc: processState?.filled ? '查看对比结果' : '先完成梳理' },
    { key: 'explore', label: '探索关联', desc: '前置/后置/相关概念' },
    { key: 'read', label: '查阅文档', desc: '查看完整说明' },
  ]

  return (
    <>
      <div className="shrink-0 px-5 pt-4 pb-3 border-b border-gray-100">
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0">
            <h2 className="text-base font-semibold text-gray-900 truncate">{concept.title}</h2>
            {concept.alias && concept.alias.length > 0 && (
              <p className="text-xs text-gray-500 mt-0.5 truncate">别名: {concept.alias.join(' / ')}</p>
            )}
          </div>
          <div className="flex items-center gap-1">
            <button onClick={onDeselect} className="shrink-0 p-1 rounded hover:bg-gray-100 text-gray-400 hover:text-gray-600" title="取消选择">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
        <div className="flex items-center gap-2 mt-2">
          <span className={`px-2 py-0.5 rounded text-xs font-medium ${
            concept.level === 1 ? 'bg-green-100 text-green-700' :
            concept.level === 2 ? 'bg-blue-100 text-blue-700' :
            'bg-purple-100 text-purple-700'
          }`}>
            L{concept.level}
          </span>
          <span className="px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-600">
            {concept.category}
          </span>
          {chain && (
            <span className="px-2 py-0.5 rounded text-xs font-medium bg-amber-100 text-amber-700">
              {chain.name}
            </span>
          )}
        </div>
      </div>

      <div className="shrink-0 flex border-b border-gray-100">
        {actions.map(a => (
          <button
            key={a.key}
            onClick={() => {
              if (a.key === 'compare' && !processState?.filled) return
              setAction(a.key)
            }}
            disabled={a.key === 'compare' && !processState?.filled}
            className={`flex-1 px-2 py-2.5 text-sm font-medium text-center transition-colors ${
              action === a.key
                ? 'text-blue-600 border-b-2 border-blue-500'
                : a.key === 'compare' && !processState?.filled
                ? 'text-gray-300 cursor-not-allowed'
                : 'text-gray-500 hover:text-gray-700'
            }`}
            title={a.desc}
          >
            <div className="text-xs">{a.label}</div>
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto">
        {action === 'process' && (
          <div className="px-5 py-8 text-center">
            <div className="p-6 rounded-lg border border-gray-200 bg-gray-50 inline-block max-w-xs">
              <div className="text-2xl mb-2">🖱️</div>
              <p className="text-sm text-gray-600 font-medium">双击图谱中的概念节点</p>
              <p className="text-xs text-gray-400 mt-1">在全屏画板中梳理推导流程</p>
            </div>
          </div>
        )}

        {action === 'compare' && (
          <ComparisonPanel
            diffs={diffs}
            userStepCount={processState?.user_flow.length ?? 0}
            referenceStepCount={referenceSteps.length}
            onNavigateGap={handleNavigateGap}
          />
        )}

        {action === 'explore' && (
          <div className="px-5 py-4 space-y-5">
            <div>
              <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">推导路径</h3>
              <div className="p-4 bg-white rounded-lg border border-gray-200">
                <DependencyChainSVG concept={concept} concepts={concepts} />
              </div>
            </div>
            {(() => {
              const related = getRelatedConcepts(concept.id, edges, concepts)
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
                        {byType.leads_to.map(({ concept: c }) => (
                          <button key={c.id} onClick={() => onNavigate(c.id)} className="w-full text-left px-3 py-2 rounded-lg border border-green-200 hover:border-green-300 hover:bg-green-50 transition-colors group">
                            <div className="flex items-center justify-between">
                              <span className="text-sm font-medium text-gray-800 group-hover:text-green-700">{c.title}</span>
                              <span className="text-xs text-gray-400">→</span>
                            </div>
                            <p className="text-xs text-gray-500 mt-0.5">{getRelationReason(concept, c, 'leads_to')}</p>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                  {byType.depends_on.length > 0 && (
                    <div>
                      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">前置基础</h3>
                      <div className="space-y-1.5">
                        {byType.depends_on.map(({ concept: c }) => (
                          <button key={c.id} onClick={() => onNavigate(c.id)} className="w-full text-left px-3 py-2 rounded-lg border border-red-200 hover:border-red-300 hover:bg-red-50 transition-colors group">
                            <div className="flex items-center justify-between">
                              <span className="text-sm font-medium text-gray-800 group-hover:text-red-700">{c.title}</span>
                              <span className="text-xs text-gray-400">←</span>
                            </div>
                            <p className="text-xs text-gray-500 mt-0.5">{getRelationReason(concept, c, 'depends_on')}</p>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                  {byType.related.length > 0 && (
                    <div>
                      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">平行关联</h3>
                      <div className="space-y-1.5">
                        {byType.related.map(({ concept: c }) => (
                          <button key={c.id} onClick={() => onNavigate(c.id)} className="w-full text-left px-3 py-2 rounded-lg border border-gray-200 hover:border-gray-300 hover:bg-gray-50 transition-colors group">
                            <div className="flex items-center justify-between">
                              <span className="text-sm font-medium text-gray-800 group-hover:text-gray-900">{c.title}</span>
                              <span className="text-xs text-gray-400">↔</span>
                            </div>
                            <p className="text-xs text-gray-500 mt-0.5">{getRelationReason(concept, c, 'related')}</p>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              )
            })()}
          </div>
        )}

        {action === 'read' && (
          <div className="px-5 py-4">
            <div className="prose prose-sm max-w-none">
              <DocumentViewer document={{
                id: concept.id, title: concept.title, path: concept.path,
                content: concept.content, level: concept.level, category: concept.category,
                tags: concept.tags, lastModified: concept.lastModified, metadata: concept.metadata,
              }} />
            </div>
          </div>
        )}
      </div>
    </>
  )
}
