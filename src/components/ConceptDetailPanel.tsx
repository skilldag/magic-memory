import { useState, useMemo, useCallback, useEffect } from 'react'
import { DocumentViewer } from './DocumentViewer'

import { ComparisonPanel } from './ComparisonPanel'
import { AlignmentPanel } from './AlignmentPanel'
import { DependencyChainSVG } from './DependencyChainSVG'
import {
  getRelatedConcepts,
  getDependencyChain,
  getRelationReason,
  getReviewRecordFor,
} from '../utils/knowledgeGraph'
import { generateReferenceFlow, diffFlows, getGapConceptIds, generateGenericChain } from '../utils/processComparison'
import { loadDocContent, clearDocCache } from '../utils/docLoader'
import { useKnowledgeGraphStore } from '../store/knowledgeGraphStore'
import { useDocumentStore } from '../store/documentStore'
import type { Concept, ConceptEdge, ReviewRecord, ProcessChain, Document } from '../types'

interface ConceptDetailPanelProps {
  concept: Concept
  concepts: Concept[]
  edges: ConceptEdge[]
  reviewRecords: Map<string, ReviewRecord>
  onNavigate: (conceptId: string) => void
  onDeselect: () => void
  onEnterProcess?: (concept: Concept) => void
}

type ActionKey = 'import' | 'compare' | 'read' | 'align'

export function ConceptDetailPanel({
  concept,
  concepts,
  edges,
  reviewRecords,
  onNavigate,
  onDeselect,
  onEnterProcess,
}: ConceptDetailPanelProps) {
  const [action, setAction] = useState<ActionKey>('read')
  const [docContent, setDocContent] = useState<string | null>(null)
  const [docLoading, setDocLoading] = useState(false)
  const chains = useKnowledgeGraphStore(s => s.chains)
  const updateConceptContent = useKnowledgeGraphStore(s => s.updateConceptContent)

  useEffect(() => {
    if (action !== 'read') return
    if (concept.content) {
      setDocContent(concept.content)
      setDocLoading(false)
      return
    }
    setDocContent(null)
    setDocLoading(true)
    const { projects, activeProjectId } = useKnowledgeGraphStore.getState()
    const project = projects.find(p => p.id === activeProjectId)
    const baseDir = project?.sourceDir || undefined
    loadDocContent(concept.path, baseDir).then(content => {
      if (content) setDocContent(content)
      setDocLoading(false)
    })
  }, [action, concept.id, concept.path, concept.content])

  const updateProcessState = useKnowledgeGraphStore(s => s.updateProcessState)
  // Questions feature removed
  const storeConcepts = useKnowledgeGraphStore(s => s.concepts)
  const addConcept = useKnowledgeGraphStore(s => s.addConcept)
  const addEdge = useKnowledgeGraphStore(s => s.addEdge)

  const [reparseStatus, setReparseStatus] = useState<'idle' | 'loading' | 'done'>('idle')
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' } | null>(null)

  const showToast = (message: string, type: 'success' | 'error') => {
    setToast({ message, type })
    setTimeout(() => setToast(null), 2500)
  }

  const handleReparseRelations = useCallback(async () => {
    const store = useKnowledgeGraphStore.getState()
    const c = store.concepts.find(c => c.id === concept.id)
    if (!c?.content) {
      showToast('请先导入文档内容', 'error')
      return
    }

    setReparseStatus('loading')

    try {
      const resp = await fetch('/api/infer-relations-from-content', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          conceptId: concept.id,
          content: c.content,
          concepts: store.concepts.map(c => ({ id: c.id, title: c.title, problem: c.problem })),
        }),
      })
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const { relations } = await resp.json()
      if (!relations || relations.length === 0) {
        setReparseStatus('idle')
        showToast('未找到相似概念，可尝试补充文档内容', 'error')
        return
      }

      const relatedIds = relations.map((r: any) => r.targetId)

      const newEdgeSet = new Set<string>()
      const newEdges: ConceptEdge[] = []

      const addEdgeToSet = (source: string, target: string, type: ConceptEdge['type']) => {
        const key = `${source}|${target}|${type}`
        if (!newEdgeSet.has(key)) {
          newEdgeSet.add(key)
          newEdges.push({ id: `${source}-${type}-${target}`, source, target, type })
        }
      }

      relatedIds.forEach((t: string) => addEdgeToSet(concept.id, t, 'related'))

      // 覆盖：保留不涉及本概念的其他边
      const keptEdges = store.edges.filter(e =>
        e.source !== concept.id && e.target !== concept.id
      )

      useKnowledgeGraphStore.setState({
        concepts: store.concepts.map(c =>
          c.id === concept.id
            ? { ...c, depends_on: c.depends_on, leads_to: c.leads_to, related: relatedIds }
            : c
        ),
        edges: [...keptEdges, ...newEdges],
      })

      setReparseStatus('done')
      showToast(`关系更新成功，关联了 ${relatedIds.length} 个概念`, 'success')
      setTimeout(() => setReparseStatus('idle'), 2500)
    } catch (e) {
      console.error('关系推断失败:', e)
      setReparseStatus('idle')
      showToast('关系推断失败，请确认后端服务已启动', 'error')
    }
  }, [concept.id])

  const handleRequestLLM = useCallback(async () => {
    try {
      const resp = await fetch('/api/generate-doc', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          concept: { title: concept.title, problem: concept.problem, gap_anticipate: concept.gap_anticipate }
        }),
      })
      if (!resp.ok) { alert('生成失败: ' + resp.statusText); return }
      const data = await resp.json()
      if (data.content) {
        const { projects, activeProjectId } = useKnowledgeGraphStore.getState()
        const project = projects.find(p => p.id === activeProjectId)
        const baseDir = project?.sourceDir || undefined
        const writeResp = await fetch('/api/write-doc', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: concept.path, content: data.content, baseDir }),
        })
        if (writeResp.ok) {
          const result = await writeResp.json()
          // 如果服务器返回了实际路径，更新概念中的 path
          if (result.filePath && result.filePath !== concept.path) {
            updateConceptContent(concept.id, data.content)
          }
          const loaded = await loadDocContent(concept.path, baseDir)
          if (loaded) setDocContent(loaded)
        }
      }
    } catch (e: any) {
      alert('生成失败: ' + (e.message || e))
    }
  }, [concept])

  const [importContent, setImportContent] = useState('')
  const [importLoading, setImportLoading] = useState(false)
  const [importError, setImportError] = useState<string | null>(null)
  const [importedContent, setImportedContent] = useState<string | null>(null)

  useEffect(() => {
    setImportContent('')
    setImportLoading(false)
    setImportError(null)
    setImportedContent(null)
  }, [concept.id])

  const handleImport = useCallback(async () => {
    if (!importContent.trim()) return
    setImportLoading(true)
    setImportError(null)
    try {
      const { projects, activeProjectId } = useKnowledgeGraphStore.getState()
      const project = projects.find(p => p.id === activeProjectId)
      const baseDir = project?.sourceDir || undefined
        const resp = await fetch('/api/write-doc', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: concept.path, content: importContent, baseDir }),
      })
      if (!resp.ok) {
        setImportError('导入失败: ' + resp.statusText)
        return
      }
      clearDocCache(concept.path, baseDir)
      updateConceptContent(concept.id, importContent)
      const updatedConcept = useKnowledgeGraphStore.getState().concepts.find(c => c.id === concept.id)
      if (updatedConcept) {
        useKnowledgeGraphStore.getState().selectConcept(updatedConcept)
      }
      const savedContent = importContent
      setImportContent('')
      setImportedContent(savedContent)
    } catch (e: any) {
      setImportError('导入失败: ' + (e.message || e))
    } finally {
      setImportLoading(false)
    }
  }, [concept.id, concept.path, importContent])

  const processState = reviewRecords.get(concept.id)?.process_state

  const chain = useMemo(() => {
    if (concept.process) {
      return chains.find(ch => ch.id === concept.process?.chain_id) ?? null
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
    { key: 'read', label: '查阅文档', desc: '查看完整说明' },
    { key: 'import', label: '导入文档', desc: '从剪贴板粘贴导入文档内容' },
    { key: 'align', label: '语义对齐', desc: '用自由文本对齐图谱，诊断知识缺口' },
    { key: 'compare', label: '对照验证', desc: processState?.filled ? '查看对比结果' : '先完成梳理' },
  ]

  return (
    <>
      {/* 通知条 */}
      {toast && (
        <div
          className={`absolute top-0 left-0 right-0 z-50 px-4 py-2 text-xs font-medium text-center shadow-sm transition-all ${
            toast.type === 'success'
              ? 'bg-green-50 text-green-700 border-b border-green-200'
              : 'bg-red-50 text-red-700 border-b border-red-200'
          }`}
        >
          {toast.message}
        </div>
      )}
      <div className="shrink-0 px-5 pt-4 pb-3 border-b border-gray-100">
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0">
            <h2 className="text-base font-semibold text-gray-900 truncate">{concept.title}</h2>
            {concept.alias && concept.alias.length > 0 && (
              <p className="text-xs text-gray-500 mt-0.5 truncate">别名: {concept.alias.join(' / ')}</p>
            )}
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
        {action === 'import' && (
          <div className="px-5 py-4">
            <div className="space-y-3">
              <p className="text-sm text-gray-600">对 LLM 使用下面的 prompt，然后将生成的内容粘贴到下方输入框中：</p>
              <div className="p-3 text-xs bg-gray-50 border border-gray-200 rounded-lg text-gray-700 whitespace-pre-wrap font-mono leading-relaxed select-all cursor-text">
{`以 Unix man page 的严谨技术风格和 markdown 的文本格式，用"问题→解决该问题的子概念和解决过程→引出下一问题"的层层推导方式，解释 ${concept.title} 的核心原理。主体用树状缩进和简洁公式，语言精炼专业,一读就懂,容易记忆,容易联想和建模,最好有例子和命名说明。在末尾添加 KEY CONCEPTS 段落, 只包含支撑理解关键问题的知识点和子概念，以空格分割。使用中文。`}
              </div>
              <textarea
                className="w-full h-48 p-3 text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-y"
                placeholder="在此粘贴 LLM 返回的文档内容..."
                value={importContent}
                onChange={e => setImportContent(e.target.value)}
              />
              <button
                onClick={handleImport}
                disabled={!importContent.trim() || importLoading}
                className="px-4 py-2 text-sm font-medium text-white bg-blue-500 rounded-lg hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
              >
                {importLoading ? '导入中...' : '导入'}
              </button>
              {importError && (
                <p className="text-sm text-red-500">{importError}</p>
              )}
              {importedContent && (
                <div className="pt-4 border-t border-gray-200">
                  <p className="text-sm font-medium text-green-600 mb-2">✓ 导入成功</p>
                  <div className="prose prose-sm max-w-none">
                    <DocumentViewer document={{
                      id: concept.id, title: concept.title, path: concept.path,
                      content: importedContent, level: concept.level, category: concept.category,
                      tags: concept.tags, lastModified: concept.lastModified, metadata: concept.metadata,
                    }} />
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {action === 'align' && (
          <AlignmentPanel
            concept={concept}
            allConcepts={concepts}
            onNavigate={onNavigate}
          />
        )}

        {action === 'compare' && (
          <ComparisonPanel
            diffs={diffs}
            userStepCount={processState?.user_flow.length ?? 0}
            referenceStepCount={referenceSteps.length}
            onNavigateGap={handleNavigateGap}
          />
        )}



        {action === 'read' && (
          <div className="px-5 py-4">
            {/* 文档头部：标题 + 更新关系按钮 */}
            {docContent && (
              <div className="flex items-center justify-between mb-3 pb-2 border-b border-gray-100">
                <span className="text-xs font-medium text-gray-400">文档内容</span>
                <button
                  onClick={handleReparseRelations}
                  disabled={reparseStatus === 'loading'}
                  className="px-2.5 py-1 text-xs rounded-md bg-gray-100 text-gray-600 hover:bg-gray-200 hover:text-gray-800 transition-colors flex items-center gap-1 disabled:opacity-50"
                >
                  <svg width={11} height={11} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  {reparseStatus === 'loading' ? '推断中...' : reparseStatus === 'done' ? '✓ 已更新' : '更新关系和图谱'}
                </button>
              </div>
            )}
            {docContent === null && !docLoading && !concept.content ? (
              <div className="flex flex-col items-center justify-center py-16 text-gray-400">
                <svg width={48} height={48} className="w-12 h-12 mb-3 text-gray-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
                <p className="text-sm mb-4">还没有对应的文档内容</p>
                <button
                  onClick={handleRequestLLM}
                  className="px-4 py-2 bg-blue-500 text-white text-sm rounded-lg hover:bg-blue-600 transition-colors"
                >
                  请求 LLM 生成
                </button>
              </div>
            ) : (
              <div className="prose prose-sm max-w-none">
                <DocumentViewer document={{
                  id: concept.id, title: concept.title, path: concept.path,
                  content: docContent ?? concept.content ?? '', level: concept.level, category: concept.category,
                  tags: concept.tags, lastModified: concept.lastModified, metadata: concept.metadata,
                }} />
              </div>
            )}
          </div>
        )}

        {/* Questions feature removed */}
      </div>
    </>
  )
}
