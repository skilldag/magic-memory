import { useEffect, useState, useRef, useMemo, useCallback } from 'react'
import { KnowledgeGraph } from './KnowledgeGraph'
import { GlobalSearch } from './GlobalSearch'
import { ConceptDetailPanel } from './ConceptDetailPanel'
import { SummaryPanel } from './SummaryPanel'
import { ProcessCanvas } from './ProcessCanvas'
import { ExploreDialog } from './ExploreDialog'
import { QuickExploreDialog } from './QuickExploreDialog'
import { ManualAddDialog } from './ManualAddDialog'
import { BatchLinkDialog } from './BatchLinkDialog'
import { AddConceptDialog } from './AddConceptDialog'
import { useKnowledgeGraphStore } from '../store/knowledgeGraphStore'
import type { ProjectInfo } from '../store/knowledgeGraphStore'
import { generateGenericChain } from '../utils/processComparison'
import { getDueConcepts } from '../utils/knowledgeGraph'
import type { Concept, SuggestionItem } from '../types'
import { useContainerSize } from '../hooks/useContainerSize'

type RelationType = 'leads_to' | 'depends_on' | 'related'

export function KnowledgeGraphView() {
  const concepts = useKnowledgeGraphStore(s => s.concepts)
  const edges = useKnowledgeGraphStore(s => s.edges)
  const loadGraph = useKnowledgeGraphStore(s => s.loadGraph)
  const isLoading = useKnowledgeGraphStore(s => s.isLoading)
  const loadingProgress = useKnowledgeGraphStore(s => s.loadingProgress)
  const reviewRecords = useKnowledgeGraphStore(s => s.reviewRecords)
  const dueConcepts = useMemo(
    () => getDueConcepts(concepts, reviewRecords).filter(d => d.badge.urgency <= 1),
    [concepts, reviewRecords]
  )
  const createConceptWithEdges = useKnowledgeGraphStore(s => s.createConceptWithEdges)
  const selectConcept = useKnowledgeGraphStore(s => s.selectConcept)
  // Skeleton and questions features removed
  // const questions = useKnowledgeGraphStore(s => s.questions)
  // const canvasHistory = useKnowledgeGraphStore(s => s.canvasHistory)
  // const skeletonCompleted = useKnowledgeGraphStore(s => s.skeletonCompleted)
  const conceptPanelMode = useKnowledgeGraphStore(s => s.conceptPanelMode)
  const setConceptPanelMode = useKnowledgeGraphStore(s => s.setConceptPanelMode)

  const conceptMastery = useKnowledgeGraphStore(s => s.conceptMastery)
  const storeSelectedConcept = useKnowledgeGraphStore(s => s.selectedConcept)
  const [selectedConceptId, setSelectedConceptId] = useState<string | null>(null)
  const selectedConcept = useMemo(() => {
    if (storeSelectedConcept) return storeSelectedConcept
    return selectedConceptId ? concepts.find(c => c.id === selectedConceptId) ?? null : null
  }, [storeSelectedConcept, selectedConceptId, concepts])

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
  const [showAddConceptDialog, setShowAddConceptDialog] = useState(false)
  const [linkMode, setLinkMode] = useState(false)
  const [linkSource, setLinkSource] = useState<string | null>(null)
  const [focusedNodeIds, setFocusedNodeIds] = useState<string[] | undefined>(undefined)
  const [relayoutKey, setRelayoutKey] = useState(0)
  const [bannerDismissed, setBannerDismissed] = useState(false)
  const { containerRef: graphContainerRef, size: containerSize } = useContainerSize<HTMLDivElement>()
  const selectedConceptRef = useRef<Concept | null>(null)
  const preventHideRef = useRef(false)
  const [rightPanelWidth, setRightPanelWidth] = useState(420)
  const containerRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => { loadGraph() }, [loadGraph])

useEffect(() => {
    const toggleHandler = () => {
      console.log('[KnowledgeGraphView] toggle-link-mode received, setting linkMode to true')
      setLinkMode(true)
    }
    const exitHandler = () => {
      console.log('[KnowledgeGraphView] exit-link-mode received, setting linkMode to false')
      setLinkMode(false)
      setLinkSource(null)
    }
    window.addEventListener('toggle-link-mode', toggleHandler)
    window.addEventListener('exit-link-mode', exitHandler)
    return () => {
      window.removeEventListener('toggle-link-mode', toggleHandler)
      window.removeEventListener('exit-link-mode', exitHandler)
    }
  }, [])
  
  useEffect(() => {
    console.log('[KnowledgeGraphView] linkMode state changed to:', linkMode)
  }, [linkMode])

  const storeSetLinkMode = useKnowledgeGraphStore(s => s.setLinkMode)
  useEffect(() => {
    storeSetLinkMode(linkMode)
  }, [linkMode, storeSetLinkMode])

  // 拖拽分割线
  const handleDividerMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    const startX = e.clientX
    const startWidth = rightPanelWidth
    const container = containerRef.current
    if (!container) return

    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'

    const handleMouseMove = (ev: MouseEvent) => {
      setRightPanelWidth(Math.max(200, startWidth + (startX - ev.clientX)))
    }

    const handleMouseUp = () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      if (selectedConceptRef.current) {
        setRelayoutKey(k => k + 1)
      }
    }

    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
  }, [rightPanelWidth])

  const handleSelectConcept = (concept: Concept) => {
    selectedConceptRef.current = concept
    setSelectedConceptId(concept.id)
    selectConcept(concept)
    cancelHideHoverActions()
    setHoverConcept(null)
    setFocusedNodeIds(undefined)
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
    ;(window as any).__store = useKnowledgeGraphStore
  }

  const handleExitProcess = () => {
    setProcessMode(false)
    setProcessConcept(null)
  }

  const chains = useKnowledgeGraphStore(s => s.chains)

  const processChain = useMemo(() => {
    if (!processConcept) return null
    if (processConcept.process) {
      return chains.find(ch => ch.id === processConcept.process?.chain_id) ?? null
    }
    return generateGenericChain(processConcept.id, concepts)
  }, [processConcept, concepts, chains])

  const handleNavigate = (conceptId: string) => {
    const concept = concepts.find(c => c.id === conceptId)
    if (concept) {
      setSelectedConceptId(concept.id)
      selectedConceptRef.current = concept
      selectConcept(concept)
      setFocusedNodeIds(undefined)
    }
  }

  const handlePathFocus = useCallback((ids: string[]) => {
    if (ids.length === 0) return
    setFocusedNodeIds(ids)
    const first = concepts.find(c => c.id === ids[0])
    if (first) {
      setSelectedConceptId(first.id)
      selectedConceptRef.current = first
      selectConcept(first)
    }
  }, [concepts, selectConcept])

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

  const projects = useKnowledgeGraphStore(s => s.projects)
  const activeProjectId = useKnowledgeGraphStore(s => s.activeProjectId)
  const fetchProjects = useKnowledgeGraphStore(s => s.fetchProjects)
  const loadProjectGraph = useKnowledgeGraphStore(s => s.loadProjectGraph)

  useEffect(() => { fetchProjects() }, [fetchProjects])

  const handleSelectProject = (projectId: string) => {
    loadProjectGraph(projectId)
  }

  const handleAddConcept = useCallback((title: string) => {
    const concept = useKnowledgeGraphStore.getState().addConcept({
      title,
      level: 1,
      category: '',
      problem: '',
      depends_on: [],
      leads_to: [],
      related: [],
      path: `./docs/user/${Date.now()}-${title.toLowerCase().replace(/\s+/g, '-')}.md`,
      tags: ['user-added'],
      metadata: { status: 'draft' as const },
    })
    setSelectedConceptId(concept.id)
    selectedConceptRef.current = concept
    useKnowledgeGraphStore.getState().selectConcept(concept)
    setShowAddConceptDialog(false)
  }, [])

  const showProjectList = concepts.length === 0 && !isLoading && !processMode && !activeProjectId
  const showEmptyGraph = activeProjectId && concepts.length === 0 && !isLoading

  return (
    <div ref={containerRef} className="flex h-full w-full overflow-hidden">
      <div ref={graphContainerRef} className="min-w-0 relative flex flex-col" style={{ width: `calc(100% - ${rightPanelWidth + 40}px)` }}>
        {loadingProgress > 0 && loadingProgress < 100 && (
          <div className="shrink-0 w-full bg-gray-100 h-1">
            <div
              className="h-full bg-blue-500 transition-all duration-300 ease-out"
              style={{ width: `${loadingProgress}%` }}
            />
          </div>
        )}
        {/* 顶部栏：搜索 + 聚焦信息 */}
        {!processMode && (
          <div className="absolute top-0 left-0 right-0 z-20 flex items-center gap-3 px-4 py-2 bg-white/80 backdrop-blur-sm border-b border-blue-200 shadow-sm">
            <GlobalSearch
              concepts={concepts}
              onSelect={(concept) => {
                handleSelectConcept(concept)
              }}
            />
            {selectedConcept && (
              <div className="flex items-center gap-2 shrink-0">
                <span className="w-2 h-2 rounded-full bg-green-500" />
                <span className="text-sm font-medium text-gray-700">
                  聚焦: <span className="text-blue-600">{selectedConcept.title}</span>
                </span>
              </div>
            )}
          </div>
        )}
        {!processMode && dueConcepts.length > 0 && !bannerDismissed && (
          <div className="shrink-0 flex items-center gap-3 px-4 py-2 bg-amber-50 border-b border-amber-200 z-10">
            <span className="text-sm">📅</span>
            <span className="text-sm text-amber-800">
              <strong>{dueConcepts.length}</strong> 个概念需要复习
              {dueConcepts[0] && ` · 最长的已过期 ${Math.abs(dueConcepts[0].daysUntilReview)} 天`}
            </span>
            <button
              onClick={() => {
                const first = dueConcepts[0]
                if (first) {
                  const concept = concepts.find(c => c.id === first.concept.id)
                  if (concept) handleSelectConcept(concept)
                }
              }}
              className="ml-auto px-3 py-1 text-xs font-medium text-amber-800 bg-amber-200/50 rounded-md hover:bg-amber-300/50 transition-colors"
            >
              查看待复习 →
            </button>
            <button
              onClick={() => setBannerDismissed(true)}
              className="w-5 h-5 flex items-center justify-center text-amber-400 hover:text-amber-600 transition-colors"
              title="关闭"
            >
              <svg width={12} height={12} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        )}
        {showProjectList ? (
          <div className="flex-1 flex flex-col items-center justify-center bg-gray-50 text-gray-500">
            <div className="max-w-md text-center space-y-6">
              <svg width={80} height={80} className="w-20 h-20 mx-auto text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
              </svg>
              <h2 className="text-lg font-medium text-gray-700">知识图谱</h2>
              <p className="text-sm text-gray-400">
                使用命令行初始化项目:
              </p>
              <div className="bg-gray-100 rounded-lg p-3 text-left text-xs font-mono text-gray-600">
                magic-memory init /path/to/docs<br />
                magic-memory server start
              </div>
              {projects.length > 0 && (
                <div className="space-y-2">
                  <p className="text-sm font-medium text-gray-500">已注册的项目:</p>
                  {projects.map(p => (
                    <button
                      key={p.id}
                      onClick={() => handleSelectProject(p.id)}
                      className="w-full px-4 py-3 bg-white border border-gray-200 rounded-lg text-left hover:border-blue-300 hover:text-blue-600 transition-colors"
                    >
                      <div className="text-sm font-medium text-gray-700">{p.name}</div>
                      <div className="text-xs text-gray-400 mt-0.5">
                        {p.conceptCount} 概念 · {p.edgeCount} 边 · {p.sourceDir}
                      </div>
                    </button>
                  ))}
                </div>
              )}
              {projects.length === 0 && (
                <p className="text-xs text-gray-400">
                  全局服务未运行或没有注册项目
                </p>
              )}
            </div>
          </div>
        ) : processMode && processConcept ? (
          <div className="flex-1 flex flex-col">
            {/* 面包屑导航（简化，移除历史栈） */}
            <div className="shrink-0 flex items-center gap-1 px-4 py-1.5 border-b border-gray-100 bg-gray-50 text-xs text-gray-500">
              <button onClick={() => { setProcessMode(false); setProcessConcept(null) }}
                className="hover:text-blue-600 transition-colors">图谱</button>
              {processConcept && (
                <span className="flex items-center gap-1">
                  <span className="text-gray-700 font-medium">{processConcept.title}</span>
                </span>
              )}
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
        ) : showEmptyGraph ? (
          <div className="flex-1 flex flex-col items-center justify-center bg-gray-50 text-gray-400 relative">
            <div className="text-center space-y-4">
              <svg width={48} height={48} className="mx-auto text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 7v10c0 2 1 3 3 3h10c2 0 3-1 3-3V7c0-2-1-3-3-3H7c-2 0-3 1-3 3z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h8M12 8v8" />
              </svg>
              <p className="text-sm">此项目暂无概念</p>
              <button
                onClick={() => setShowAddConceptDialog(true)}
                className="px-4 py-2 bg-blue-500 text-white text-sm font-medium rounded-lg hover:bg-blue-600 transition-colors"
              >
                + 添加概念
              </button>
              <p className="text-xs text-gray-300">或使用命令行从文档构建: <code className="font-mono">memo init /path/to/docs</code></p>
            </div>
          </div>
        ) : (
          <KnowledgeGraph
            concepts={concepts} edges={edges} selectedConcept={selectedConcept}
            focusEnabled={true}
            focusedNodeIds={focusedNodeIds}
            conceptMastery={conceptMastery}
            containerWidth={containerSize?.width}
            containerHeight={containerSize?.height}
            relayoutKey={selectedConcept ? relayoutKey : 0}
            reviewRecords={reviewRecords}
            linkMode={linkMode}
            linkSource={linkSource}
            onSelectConcept={handleSelectConcept} onNavigate={handleNavigate}
            onDoubleTapConcept={(c) => { handleSelectConcept(c); handleEnterProcess(c) }}
            onBackgroundDoubleTap={() => { setShowAddConceptDialog(true) }}
            onHoverConcept={payload => { cancelHideHoverActions(); setHoverConcept(payload) }}
            onHoverLeave={() => scheduleHideHoverActions()}
            onToggleLinkMode={() => { 
              setLinkMode(!linkMode)
              if (!linkMode) setLinkSource(null) 
            }}
            onLinkStart={(id) => setLinkSource(id)}
            onLinkEnd={(sourceId, targetId) => {
              const addEdge = useKnowledgeGraphStore.getState().addEdge
              addEdge(sourceId, targetId, 'related')
              setLinkSource(null)
              setLinkMode(false)
            }}
            onLinkCancel={() => { setLinkSource(null); setLinkMode(false) }}
            onExitFocus={() => {
              setSelectedConceptId(null)
              useKnowledgeGraphStore.setState({ selectedConcept: null })
            }}
            onDeleteEdge={(edgeId) => {
              useKnowledgeGraphStore.getState().removeEdge(edgeId)
            }}
          />
        )}
        {hoverConcept && selectedConcept && (
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
            <button type="button" onMouseDown={e => e.preventDefault()} onClick={async () => {
              const c = hoverConcept.concept
              setHoverConcept(null)
              try {
                await fetch('/api/delete-doc', {
                  method: 'DELETE',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ path: c.path }),
                })
              } catch {}
              useKnowledgeGraphStore.getState().removeConcept(c.id)
            }}
              className="absolute flex items-center justify-center rounded-full bg-gradient-to-br from-red-500 to-red-700 text-white shadow-lg hover:shadow-xl hover:from-red-600 hover:to-red-800 transition-all cursor-pointer select-none font-bold"
              style={{ width: 28, height: 28, fontSize: 11, left: hoverConcept.x + (Math.max(hoverConcept.width, hoverConcept.height) / 2 + 2) * Math.cos(-90 * Math.PI / 180) - 14, top: hoverConcept.y + (Math.max(hoverConcept.width, hoverConcept.height) / 2 + 2) * Math.sin(-90 * Math.PI / 180) - 14, pointerEvents: 'auto' }} title="删除概念">删</button>
          </div>
        )}
      </div>

      {/* 可拖拽分割线 */}
      <div
        style={{
          width: 40,
          cursor: 'col-resize',
          backgroundColor: 'rgba(59, 130, 246, 0.15)',
          flexShrink: 0,
          position: 'relative',
          zIndex: 9999,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          userSelect: 'none',
        }}
        onMouseDown={handleDividerMouseDown}
      >
        <div style={{
          width: 4,
          height: 48,
          borderRadius: 2,
          backgroundColor: 'rgba(59, 130, 246, 0.5)',
          pointerEvents: 'none',
        }} />
      </div>

      {/* 右侧面板 */}
      <div className="shrink-0 bg-white flex flex-col overflow-hidden border-l border-gray-200" style={{ width: rightPanelWidth }}>
        {selectedConcept ? (
          <ConceptDetailPanel
            concept={selectedConcept} concepts={concepts} edges={edges}
            reviewRecords={reviewRecords}
            onNavigate={handleNavigate} onDeselect={() => { setSelectedConceptId(null); useKnowledgeGraphStore.setState({ selectedConcept: null }) }}
            onEnterProcess={handleEnterProcess}
          />
        ) : (
          <SummaryPanel
            onNavigate={handleNavigate}
            onPathFocus={handlePathFocus}
          />
        )}
      </div>

      {/* Question dialog removed */}
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
      {showAddConceptDialog && (
        <AddConceptDialog
          onClose={() => setShowAddConceptDialog(false)}
          onConfirm={handleAddConcept}
        />
      )}
    </div>
  )
}
