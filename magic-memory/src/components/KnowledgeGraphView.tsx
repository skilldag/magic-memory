import { useEffect, useState, useRef, useMemo } from 'react'
import { KnowledgeGraph } from './KnowledgeGraph'
import { ConceptDetailPanel } from './ConceptDetailPanel'
import { ProcessCanvas } from './ProcessCanvas'
import { ExploreDialog } from './ExploreDialog'
import { QuickExploreDialog } from './QuickExploreDialog'
import { ManualAddDialog } from './ManualAddDialog'
import { BatchLinkDialog } from './BatchLinkDialog'
import { useKnowledgeGraphStore } from '../store/knowledgeGraphStore'
import { generateGenericChain } from '../utils/processComparison'
import type { Concept, SuggestionItem } from '../types'

type RelationType = 'leads_to' | 'depends_on' | 'related'

export function KnowledgeGraphView() {
  const concepts = useKnowledgeGraphStore(s => s.concepts)
  const edges = useKnowledgeGraphStore(s => s.edges)
  const loadGraph = useKnowledgeGraphStore(s => s.loadGraph)
  const reviewRecords = useKnowledgeGraphStore(s => s.reviewRecords)
  const createConceptWithEdges = useKnowledgeGraphStore(s => s.createConceptWithEdges)

  const [selectedConcept, setSelectedConcept] = useState<Concept | null>(null)
  const [processMode, setProcessMode] = useState(false)
  const [processConcept, setProcessConcept] = useState<Concept | null>(null)
  const [hoverConcept, setHoverConcept] = useState<{ concept: Concept; x: number; y: number; width: number; height: number } | null>(null)
  const [actionConcept, setActionConcept] = useState<Concept | null>(null)
  const [hideHoverTimer, setHideHoverTimer] = useState<number | null>(null)
  const [showManualLinkDialog, setShowManualLinkDialog] = useState(false)
  const [showBatchLinkDialog, setShowBatchLinkDialog] = useState(false)
  const [batchSuggestions, setBatchSuggestions] = useState<SuggestionItem[]>([])
  const [batchLoading, setBatchLoading] = useState(false)
  const [showExploreDialog, setShowExploreDialog] = useState(false)
  const [showQuickExploreDialog, setShowQuickExploreDialog] = useState(false)
  const graphContainerRef = useRef<HTMLDivElement>(null)
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 })
  const selectedConceptRef = useRef<Concept | null>(null)
  const preventHideRef = useRef(false)

  useEffect(() => { loadGraph() }, [loadGraph])

  useEffect(() => {
    const el = graphContainerRef.current
    if (!el) return
    const ro = new ResizeObserver(entries => {
      for (const entry of entries) {
        setContainerSize({ width: entry.contentRect.width, height: entry.contentRect.height })
      }
    })
    ro.observe(el)
    setContainerSize({ width: el.clientWidth, height: el.clientHeight })
    return () => ro.disconnect()
  }, [])

  const handleSelectConcept = (concept: Concept) => {
    selectedConceptRef.current = concept
    setSelectedConcept(concept)
    cancelHideHoverActions()
    setHoverConcept(null)
  }

  const handleEnterProcess = (concept: Concept) => {
    setProcessConcept(concept)
    setProcessMode(true)
  }

  // Expose for debugging
  if (typeof window !== 'undefined') {
    (window as any).__openProcessCanvas = (conceptId: string) => {
      const c = concepts.find(c => c.id === conceptId)
      if (c) handleEnterProcess(c)
    }
  }

  const handleExitProcess = () => {
    setProcessMode(false)
    setProcessConcept(null)
  }

  const chains = useKnowledgeGraphStore(s => s.chains)

  const processChain = useMemo(() => {
    if (!processConcept) return null
    if (processConcept.process) {
      return chains.find(ch => ch.id === processConcept.process.chain_id) ?? null
    }
    return generateGenericChain(processConcept.id, concepts)
  }, [processConcept, concepts, chains])

  const handleNavigate = (conceptId: string) => {
    const concept = concepts.find(c => c.id === conceptId)
    if (concept) {
      setSelectedConcept(concept)
      selectedConceptRef.current = concept
    }
  }

  const handleManualAdd = (source: Concept, titles: string[], relationType: RelationType) => {
    titles.forEach(title => {
      createConceptWithEdges(source, { title, problem: '', relationType })
    })
    setShowManualLinkDialog(false)
  }

  const generateBatchSuggestions = async (source: Concept) => {
    setBatchLoading(true)
    const prompts = [
      `与${source.title}相关的核心概念是什么？请返回一个具体的概念名称。`,
      `${source.title}的前置基础知识中最重要的概念是什么？`,
      `${source.title}的延伸或进阶概念是什么？`,
    ]
    const results: SuggestionItem[] = []

    for (const p of prompts) {
      try {
        const resp = await fetch('http://localhost:4321/api/explore', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            sourceConcept: { id: source.id, title: source.title, problem: source.problem },
            userQuestion: p,
            relationType: 'related',
          }),
        })
        if (!resp.ok) continue
        const data = await resp.json()
        if (data?.title) {
          results.push({ title: data.title, problem: data.problem || `与${source.title}相关`, relationType: 'related', checked: true })
        }
      } catch { /* skip */ }
    }

    if (results.length === 0) {
      setBatchSuggestions([])
      setBatchLoading(false)
      return
    }

    const existingTitles = new Set(concepts.map(c => c.title.toLowerCase()))
    const dedup = Array.from(new Map(results.map(r => [r.title, r])).values())
      .filter(r => !existingTitles.has(r.title.toLowerCase()))
      .slice(0, 8)
    setBatchSuggestions(dedup)
    setBatchLoading(false)
  }

  const confirmBatchAdd = () => {
    if (!actionConcept) return
    batchSuggestions.filter(s => s.checked).forEach(s => {
      createConceptWithEdges(actionConcept, { title: s.title, problem: s.problem, relationType: s.relationType })
    })
    setShowBatchLinkDialog(false)
    setBatchSuggestions([])
  }

  const cancelHideHoverActions = () => {
    if (hideHoverTimer) { window.clearTimeout(hideHoverTimer); setHideHoverTimer(null) }
  }

  const scheduleHideHoverActions = () => {
    if (preventHideRef.current) return
    cancelHideHoverActions()
    const timerId = window.setTimeout(() => {
      if (!showManualLinkDialog && !showBatchLinkDialog && !showQuickExploreDialog) setHoverConcept(null)
    }, 300)
    setHideHoverTimer(timerId)
  }

  return (
    <div className="flex h-full w-full overflow-hidden">
      {/* 左侧图谱 / 过程画板 */}
      <div ref={graphContainerRef} className="flex-1 min-w-0 relative flex flex-col">
        {processMode && processConcept ? (
          <div className="flex-1 flex flex-col">
            <div className="shrink-0 flex items-center gap-2 px-4 py-2 border-b border-gray-200 bg-white">
              <button
                onClick={handleExitProcess}
                className="flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-gray-600 bg-gray-100 rounded-md hover:bg-gray-200 transition-colors"
              >
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
                返回图谱
              </button>
              <span className="text-sm font-medium text-gray-700">{processConcept.title}</span>
              <span className="text-xs text-gray-400">过程梳理画板</span>
            </div>
            <div className="flex-1">
              <ProcessCanvas
                concept={processConcept}
                chain={processChain}
                allConcepts={concepts}
                onComplete={(flow) => {
                  useKnowledgeGraphStore.getState().updateProcessState(processConcept.id, {
                    user_flow: flow,
                    filled: true,
                    compared: false,
                  })
                }}
                onNavigate={handleNavigate}
              />
            </div>
          </div>
        ) : (
          <KnowledgeGraph
            concepts={concepts} edges={edges} selectedConcept={selectedConcept}
            focusEnabled={true}
            onSelectConcept={handleSelectConcept} onNavigate={handleNavigate}
            onDoubleTapConcept={(c) => { handleSelectConcept(c); handleEnterProcess(c) }}
            onBackgroundDoubleTap={() => { setSelectedConcept(null); selectedConceptRef.current = null; cancelHideHoverActions(); setHoverConcept(null) }}
            onHoverConcept={payload => { cancelHideHoverActions(); setHoverConcept(payload) }}
            onHoverLeave={() => scheduleHideHoverActions()}
          />
        )}
        {hoverConcept && selectedConceptRef.current && (
          <div className="absolute z-30" style={{ left: 0, top: 0, pointerEvents: 'none' }}>
            <button type="button" onMouseDown={e => e.preventDefault()} onClick={() => { setActionConcept(hoverConcept.concept); setShowBatchLinkDialog(true); setHoverConcept(null); void generateBatchSuggestions(hoverConcept.concept) }}
              className="absolute flex items-center justify-center rounded-full bg-gradient-to-br from-blue-500 to-blue-700 text-white shadow-lg hover:shadow-xl hover:from-blue-600 hover:to-blue-800 transition-all cursor-pointer select-none font-bold"
              style={{ width: 28, height: 28, fontSize: 11, left: hoverConcept.x + Math.max(hoverConcept.width, hoverConcept.height) / 2 + 2 - 14, top: hoverConcept.y - 14, pointerEvents: 'auto' }} title="AI 生成探索">AI</button>
            <button type="button" onMouseDown={e => e.preventDefault()} onClick={() => { setActionConcept(hoverConcept.concept); setShowQuickExploreDialog(true); setHoverConcept(null) }}
              className="absolute flex items-center justify-center rounded-full bg-gradient-to-br from-purple-400 to-violet-500 text-white shadow-lg hover:shadow-xl hover:from-purple-500 hover:to-violet-600 transition-all cursor-pointer select-none font-bold"
              style={{ width: 28, height: 28, fontSize: 11, left: hoverConcept.x + (Math.max(hoverConcept.width, hoverConcept.height) / 2 + 2) * Math.cos(-35 * Math.PI / 180) - 14, top: hoverConcept.y + (Math.max(hoverConcept.width, hoverConcept.height) / 2 + 2) * Math.sin(-35 * Math.PI / 180) - 14, pointerEvents: 'auto' }} title="基于问题生成概念">?</button>
            <button type="button" onMouseDown={e => e.preventDefault()} onClick={() => { setActionConcept(hoverConcept.concept); setShowManualLinkDialog(true); setHoverConcept(null) }}
              className="absolute flex items-center justify-center rounded-full bg-gradient-to-br from-green-400 to-teal-500 text-white shadow-lg hover:shadow-xl hover:from-green-500 hover:to-teal-600 transition-all cursor-pointer select-none font-bold"
              style={{ width: 28, height: 28, fontSize: 9, left: hoverConcept.x + (Math.max(hoverConcept.width, hoverConcept.height) / 2 + 2) * Math.cos(35 * Math.PI / 180) - 14, top: hoverConcept.y + (Math.max(hoverConcept.width, hoverConcept.height) / 2 + 2) * Math.sin(35 * Math.PI / 180) - 14, pointerEvents: 'auto' }} title="手动添加概念">手动</button>
          </div>
        )}
      </div>

      {/* 右侧面板 */}
      <div className="w-[420px] xl:w-[480px] shrink-0 border-l border-gray-200 bg-white flex flex-col overflow-hidden">
        {selectedConcept ? (
          <ConceptDetailPanel
            concept={selectedConcept} concepts={concepts} edges={edges}
            reviewRecords={reviewRecords}
            onNavigate={handleNavigate} onDeselect={() => setSelectedConcept(null)}
            onEnterProcess={handleEnterProcess}
          />
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center text-gray-400 px-8">
            <svg className="w-16 h-16 mb-4 text-gray-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
            <p className="text-sm text-center leading-relaxed">点击图谱中的节点<br/>查看概念详情</p>
          </div>
        )}
      </div>

      {showExploreDialog && actionConcept && (
        <ExploreDialog sourceConcept={actionConcept} onClose={() => { setShowExploreDialog(false); setHoverConcept(null) }} />
      )}
      {showQuickExploreDialog && actionConcept && (
        <QuickExploreDialog sourceConcept={actionConcept} onClose={() => { setShowQuickExploreDialog(false); setHoverConcept(null) }} />
      )}
      {showManualLinkDialog && actionConcept && (
        <ManualAddDialog sourceConcept={actionConcept} onClose={() => setShowManualLinkDialog(false)} onAdd={handleManualAdd} />
      )}
      {showBatchLinkDialog && actionConcept && (
        <BatchLinkDialog sourceConcept={actionConcept} suggestions={batchSuggestions} loading={batchLoading}
          onClose={() => { setShowBatchLinkDialog(false); setBatchSuggestions([]) }}
          onToggle={(idx) => setBatchSuggestions(prev => prev.map((v, i) => i === idx ? { ...v, checked: !v.checked } : v))}
          onConfirm={confirmBatchAdd}
        />
      )}
    </div>
  )
}
