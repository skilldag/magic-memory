import { useEffect, useState, useRef } from 'react'
import { KnowledgeGraph } from './KnowledgeGraph'
import { ConceptDetailPanel } from './ConceptDetailPanel'
import { ExploreDialog } from './ExploreDialog'
import { ManualAddDialog } from './ManualAddDialog'
import { BatchLinkDialog } from './BatchLinkDialog'
import { useKnowledgeGraphStore } from '../store/knowledgeGraphStore'
import type { Concept, SuggestionItem } from '../types'

type RelationType = 'leads_to' | 'depends_on' | 'related'

export function KnowledgeGraphView() {
  const concepts = useKnowledgeGraphStore(s => s.concepts)
  const edges = useKnowledgeGraphStore(s => s.edges)
  const viewMode = useKnowledgeGraphStore(s => s.viewMode)
  const setViewMode = useKnowledgeGraphStore(s => s.setViewMode)
  const loadGraph = useKnowledgeGraphStore(s => s.loadGraph)
  const startReview = useKnowledgeGraphStore(s => s.startReview)
  const reviewRecords = useKnowledgeGraphStore(s => s.reviewRecords)
  const createConceptWithEdges = useKnowledgeGraphStore(s => s.createConceptWithEdges)

  const [selectedConcept, setSelectedConcept] = useState<Concept | null>(null)
  const [lastReviewFeedback, setLastReviewFeedback] = useState<string | null>(null)
  const [hoverConcept, setHoverConcept] = useState<{ concept: Concept; x: number; y: number; width: number; height: number } | null>(null)
  const [actionConcept, setActionConcept] = useState<Concept | null>(null)
  const [hideHoverTimer, setHideHoverTimer] = useState<number | null>(null)
  const [showManualLinkDialog, setShowManualLinkDialog] = useState(false)
  const [showBatchLinkDialog, setShowBatchLinkDialog] = useState(false)
  const [batchSuggestions, setBatchSuggestions] = useState<SuggestionItem[]>([])
  const [batchLoading, setBatchLoading] = useState(false)
  const [showExploreDialog, setShowExploreDialog] = useState(false)
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

  const handleNavigate = (conceptId: string) => {
    const concept = concepts.find(c => c.id === conceptId)
    if (concept) {
      setSelectedConcept(concept)
      selectedConceptRef.current = concept
    }
  }

  const handleReviewScore = (quality: number) => {
    if (!selectedConcept) return
    startReview(selectedConcept.id, quality)
    const msgs: Record<number, string> = {
      2: '已记录：这题偏难，建议先回看前置基础再复习。',
      3: '已记录：基本掌握，建议今天再做一轮巩固。',
      5: '已记录：掌握较好，系统会适当拉长下次复习间隔。',
    }
    setLastReviewFeedback(msgs[quality] || '已记录')
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
      if (!showManualLinkDialog && !showBatchLinkDialog) setHoverConcept(null)
    }, 300)
    setHideHoverTimer(timerId)
  }

  return (
    <div className="flex h-full w-full overflow-hidden">
      {/* 左侧图谱 */}
      <div ref={graphContainerRef} className="flex-1 min-w-0 relative">
        <div className="absolute top-3 left-3 z-10 pointer-events-auto flex items-center gap-2 bg-white/90 backdrop-blur rounded-lg shadow px-1.5 py-1">
          {[{ key: 'explore' as const, label: '探索' }, { key: 'review' as const, label: '学习' }].map(b => (
            <button key={b.key} onClick={() => { setViewMode(b.key); setHoverConcept(null) }}
              className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${viewMode === b.key ? 'bg-blue-600 text-white border border-blue-700 shadow' : 'bg-white text-gray-600 border border-transparent hover:bg-gray-100'}`}>
              {b.label}
            </button>
          ))}
        </div>
        <KnowledgeGraph
          concepts={concepts} edges={edges} selectedConcept={selectedConcept}
          focusEnabled={viewMode === 'explore'}
          onSelectConcept={handleSelectConcept} onNavigate={handleNavigate}
          onBackgroundDoubleTap={() => { setSelectedConcept(null); selectedConceptRef.current = null; cancelHideHoverActions(); setHoverConcept(null) }}
          onHoverConcept={payload => { cancelHideHoverActions(); setHoverConcept(payload) }}
          onHoverLeave={() => scheduleHideHoverActions()}
        />
        {hoverConcept && viewMode === 'explore' && selectedConceptRef.current && (
          <div className="absolute z-30" style={{ left: 0, top: 0, pointerEvents: 'none' }}>
            <button type="button" onMouseDown={e => e.preventDefault()} onClick={() => { setActionConcept(hoverConcept.concept); setShowBatchLinkDialog(true); setHoverConcept(null); void generateBatchSuggestions(hoverConcept.concept) }}
              className="absolute flex items-center justify-center rounded-full bg-gradient-to-br from-blue-500 to-blue-700 text-white shadow-lg hover:shadow-xl hover:from-blue-600 hover:to-blue-800 transition-all cursor-pointer select-none font-bold"
              style={{ width: 28, height: 28, fontSize: 11, left: hoverConcept.x + Math.max(hoverConcept.width, hoverConcept.height) / 2 + 2 - 14, top: hoverConcept.y - 14, pointerEvents: 'auto' }} title="AI 生成探索">AI</button>
            <button type="button" onMouseDown={e => e.preventDefault()} onClick={() => { setActionConcept(hoverConcept.concept); setShowManualLinkDialog(true); setHoverConcept(null) }}
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
            concept={selectedConcept} viewMode={viewMode} concepts={concepts} edges={edges}
            reviewRecords={reviewRecords} lastReviewFeedback={lastReviewFeedback}
            onNavigate={handleNavigate} onDeselect={() => setSelectedConcept(null)}
            onReviewScore={handleReviewScore}
            onOpenExplore={() => { setActionConcept(selectedConcept); setShowExploreDialog(true) }}
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
