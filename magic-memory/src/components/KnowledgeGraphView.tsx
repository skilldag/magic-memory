import { useEffect, useState, useRef, useMemo, useCallback } from 'react'
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
import { parseFrontmatter, matchTitlesToIds } from '../utils/conceptParser'

type RelationType = 'leads_to' | 'depends_on' | 'related'

export function KnowledgeGraphView() {
  const concepts = useKnowledgeGraphStore(s => s.concepts)
  const edges = useKnowledgeGraphStore(s => s.edges)
  const loadGraph = useKnowledgeGraphStore(s => s.loadGraph)
  const isLoading = useKnowledgeGraphStore(s => s.isLoading)
  const reviewRecords = useKnowledgeGraphStore(s => s.reviewRecords)
  const createConceptWithEdges = useKnowledgeGraphStore(s => s.createConceptWithEdges)
  // Skeleton and questions features removed
  // const questions = useKnowledgeGraphStore(s => s.questions)
  // const canvasHistory = useKnowledgeGraphStore(s => s.canvasHistory)
  // const skeletonCompleted = useKnowledgeGraphStore(s => s.skeletonCompleted)
  const conceptPanelMode = useKnowledgeGraphStore(s => s.conceptPanelMode)
  const setConceptPanelMode = useKnowledgeGraphStore(s => s.setConceptPanelMode)

  const [selectedConceptId, setSelectedConceptId] = useState<string | null>(null)
  const selectedConcept = useMemo(() =>
    selectedConceptId ? concepts.find(c => c.id === selectedConceptId) ?? null : null,
    [selectedConceptId, concepts]
  )
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
  // Question dialog removed
  const [folderHandle, setFolderHandle] = useState<FileSystemDirectoryHandle | null>(null)
  const [folderName, setFolderName] = useState('')
  const [isScanning, setIsScanning] = useState(false)
  const graphContainerRef = useRef<HTMLDivElement>(null)
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 })
  const selectedConceptRef = useRef<Concept | null>(null)
  const preventHideRef = useRef(false)
  const [rightPanelWidth, setRightPanelWidth] = useState(420)
  const isResizing = useRef(false)
  const startXRef = useRef(0)
  const startWidthRef = useRef(0)
  const containerRef = useRef<HTMLDivElement>(null)

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

  // 拖拽分割线
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing.current || !containerRef.current) return
      const containerRect = containerRef.current.getBoundingClientRect()
      const maxWidth = Math.min(containerRect.width * 0.6, 720)
      const newWidth = Math.max(300, Math.min(maxWidth, startWidthRef.current + (startXRef.current - e.clientX)))
      setRightPanelWidth(newWidth)
    }
    const handleMouseUp = () => {
      isResizing.current = false
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
  }, [])

  const handleDividerMouseDown = (e: React.MouseEvent) => {
    e.preventDefault()
    isResizing.current = true
    startXRef.current = e.clientX
    startWidthRef.current = rightPanelWidth
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }

  const handleSelectConcept = (concept: Concept) => {
    selectedConceptRef.current = concept
    setSelectedConceptId(concept.id)
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
      setSelectedConceptId(concept.id)
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

  // Skeleton mode removed: skeletonNodes no longer used

  const handleBrowseFolder = useCallback(async () => {
    try {
      const handle = await (window as any).showDirectoryPicker()
      setFolderHandle(handle)
      setFolderName(handle.name)
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        alert('选择文件夹失败: ' + (e.message || e))
      }
    }
  }, [])

  // 递归读取目录下所有 .md 文件内容
  async function readMdFiles(dirHandle: FileSystemDirectoryHandle, pathPrefix = ''): Promise<{ path: string; content: string }[]> {
    const results: { path: string; content: string }[] = []
    for await (const [name, entry] of (dirHandle as any).entries()) {
      const entryPath = pathPrefix ? `${pathPrefix}/${name}` : name
      if (entry.kind === 'directory' && !name.startsWith('.')) {
        results.push(...await readMdFiles(entry, entryPath))
      } else if (entry.kind === 'file' && name.endsWith('.md')) {
        try {
          const file = await (entry as FileSystemFileHandle).getFile()
          const content = await file.text()
          if (content.trim()) {
            results.push({ path: entryPath, content })
          }
        } catch { /* skip unreadable */ }
      }
    }
    return results
  }

  const handleAutoScan = useCallback(async () => {
    if (!folderHandle) { alert('请先选择文档目录'); return }
    setIsScanning(true)
    try {
      const files = await readMdFiles(folderHandle)
      const concepts: any[] = []
      
      for (const file of files) {
        const parsed = parseFrontmatter(file.content)
        const hasFm = parsed.meta && Object.keys(parsed.meta).length > 0
        
        if (hasFm) {
          // 有 frontmatter → 直接提取
          const meta = parsed.meta as any
          concepts.push({
            id: meta.id || file.path.replace('.md', '').replace(/\//g, '-'),
            title: meta.title || file.path.replace('.md', ''),
            path: file.path,
            level: meta.level ?? 1,
            category: meta.category || '',
            problem: meta.problem || '',
            gap_anticipate: meta.gap_anticipate || '',
            depends_on: meta.depends_on || [],
            leads_to: meta.leads_to || [],
            related: meta.related || [],
            alias: meta.alias,
            tags: meta.tags || [],
          })
        } else {
          // 无 frontmatter → 调 LLM 推断
          try {
            const resp = await fetch('/api/infer-frontmatter', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ filename: file.path, content: file.content.slice(0, 3000) }),
            })
            if (!resp.ok) continue
            const llmResult = await resp.json()
            if (!llmResult.title) continue
            
            concepts.push({
              id: llmResult.id || file.path.replace('.md', '').replace(/\//g, '-'),
              title: llmResult.title,
              path: file.path,
              level: llmResult.level ?? 1,
              category: llmResult.category || '',
              problem: llmResult.problem || '',
              gap_anticipate: llmResult.gap_anticipate || '',
              depends_on: llmResult.depends_on_titles || [],
              leads_to: llmResult.leads_to_titles || [],
              related: llmResult.related_titles || [],
              alias: llmResult.alias,
              elements: llmResult.elements || [],
              tags: llmResult.tags || [],
            })
          } catch { /* skip LLM failures */ }
        }
      }

      // 用 matchTitlesToIds 将标题引用转为 ID
      const built = concepts.map(c => ({
        ...c,
        depends_on: matchTitlesToIds(c.depends_on, concepts),
        leads_to: matchTitlesToIds(c.leads_to, concepts),
        related: matchTitlesToIds(c.related, concepts),
      }))

      // 推导边（处理全部三种关系类型）
      const ids = new Set(built.map(c => c.id))
      const edges: any[] = []
      const edgeSet = new Set<string>()
      for (const c of built) {
        for (const t of c.leads_to) {
          if (ids.has(t)) {
            const eid = `${c.id}-leads-${t}`
            if (!edgeSet.has(eid)) { edgeSet.add(eid); edges.push({ id: eid, source: c.id, target: t, type: 'leads_to' }) }
          }
        }
        for (const t of c.depends_on) {
          if (ids.has(t)) {
            const eid = `${c.id}-depends-${t}`
            if (!edgeSet.has(eid)) { edgeSet.add(eid); edges.push({ id: eid, source: c.id, target: t, type: 'depends_on' }) }
          }
        }
        for (const t of c.related) {
          if (ids.has(t)) {
            const eid = `${c.id}-related-${t}`
            if (!edgeSet.has(eid)) { edgeSet.add(eid); edges.push({ id: eid, source: c.id, target: t, type: 'related' }) }
          }
        }
      }
      
      if (built.length > 0) {
        useKnowledgeGraphStore.setState({ concepts: built, edges, isLoading: false })
      } else {
        alert('所选目录中没有找到可识别的概念文档')
      }
    } catch (e: any) {
      alert('扫描失败: ' + (e.message || e))
    } finally {
      setIsScanning(false)
    }
  }, [folderHandle])

  const handleOnboardingManualAdd = useCallback(() => {
    if (!folderHandle) { alert('请先选择文档目录'); return }
    setShowQuickExploreDialog(true)
  }, [folderHandle])

  // 检测是否显示空状态引导
  const isEmpty = concepts.length === 0 && !isLoading && !processMode

  return (
    <div ref={containerRef} className="flex h-full w-full overflow-hidden">
      {/* 左侧图谱 / 过程画板 */}
      <div ref={graphContainerRef} className="flex-1 min-w-0 relative flex flex-col">
        {isEmpty ? (
          <div className="flex-1 flex flex-col items-center justify-center bg-gray-50 text-gray-500">
            <div className="max-w-md text-center space-y-6">
              <svg className="w-20 h-20 mx-auto text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
              </svg>
              <h2 className="text-lg font-medium text-gray-700">还没有知识图谱索引</h2>
              <p className="text-sm text-gray-400">设置文档目录路径，选择索引生成方式</p>
              <div className="space-y-4">
                <button
                  onClick={handleBrowseFolder}
                  className="w-full px-4 py-3 border-2 border-dashed border-gray-300 rounded-lg text-sm text-gray-500 hover:border-blue-400 hover:text-blue-500 transition-colors"
                >
                  {folderName ? `📂 ${folderName}` : '点击选择文档目录'}
                </button>
                {folderName && (
                  <div className="flex gap-3">
                    <button
                      onClick={handleAutoScan}
                      disabled={isScanning}
                      className="flex-1 px-4 py-2.5 bg-blue-500 text-white text-sm font-medium rounded-lg hover:bg-blue-600 disabled:bg-blue-300 transition-colors"
                    >
                      {isScanning ? '扫描中...' : '自动扫描建索引'}
                    </button>
                    <button
                      onClick={handleOnboardingManualAdd}
                      className="flex-1 px-4 py-2.5 bg-white text-gray-700 text-sm font-medium rounded-lg border border-gray-200 hover:border-blue-300 hover:text-blue-600 transition-colors"
                    >
                      手动添加概念
                    </button>
                  </div>
                )}
              </div>
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
        ) : (
          <KnowledgeGraph
            concepts={concepts} edges={edges} selectedConcept={selectedConcept}
            focusEnabled={true}
            onSelectConcept={handleSelectConcept} onNavigate={handleNavigate}
            onDoubleTapConcept={(c) => { handleSelectConcept(c); handleEnterProcess(c) }}
            onBackgroundDoubleTap={() => { setSelectedConceptId(null); selectedConceptRef.current = null; cancelHideHoverActions(); setHoverConcept(null) }}
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

      {/* 可拖拽分割线 */}
      <div
        className="w-1 shrink-0 cursor-col-resize bg-transparent hover:bg-blue-300 active:bg-blue-400 transition-colors relative z-10"
        onMouseDown={handleDividerMouseDown}
      />

      {/* 右侧面板 */}
      <div className="shrink-0 bg-white flex flex-col overflow-hidden border-l border-gray-200" style={{ width: rightPanelWidth }}>
        {selectedConcept ? (
          <ConceptDetailPanel
            concept={selectedConcept} concepts={concepts} edges={edges}
            reviewRecords={reviewRecords}
            onNavigate={handleNavigate} onDeselect={() => setSelectedConceptId(null)}
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
    </div>
  )
}
